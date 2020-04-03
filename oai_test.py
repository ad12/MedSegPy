# Author: Zhongnan Fang, zhongnanf@gmail.com, 2017 July
# Modified: Akshay Chaudhari, akshaysc@stanford.edu 2017 August
#           Arjun Desai, arjundd@stanford.edu, 2018 June

import logging
import matplotlib

matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import argparse
import os
import time
from time import strptime, strftime
import h5py

import numpy as np
import scipy.io as sio
from keras import backend as K

import mri_utils
import utils.utils as utils
from utils import io_utils, im_utils, dl_utils
from utils.metric_utils import MetricsManager
from utils.im_utils import MultiClassOverlay


import config as MCONFIG
from config import DeeplabV3Config, SegnetConfig, UNetConfig, UNet2_5DConfig, ResidualUNet
from utils.metric_utils import dice_score_coefficient
from utils.logger import setup_logger
from models.models import get_model
from keras.utils import plot_model
from scan_metadata import ScanMetadata
from generators.im_gens import get_generator
from stat import S_IREAD, S_IRGRP, S_IROTH

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))

DATE_THRESHOLD = strptime('2018-09-01-22-39-39', '%Y-%m-%d-%H-%M-%S')
TEST_SET_METADATA_PIK = '/bmrNAS/people/arjun/msk_seg_networks/oai_metadata/oai_data.dat'
TEST_SET_MD = io_utils.load_pik(TEST_SET_METADATA_PIK)

VOXEL_SPACING = (0.3125, 0.3125, 0.7)
SAVE_H5_DATA = False


def get_voxel_spacing(num_slices):
    if num_slices == 72:
        return (0.3125, 0.3125, 1.4)
    elif num_slices == 160:
        return (0.3125, 0.3125, 0.7)
    raise ValueError('num_slices %d unknown' % num_slices)


def find_start_and_end_slice(y_true):
    for i in range(y_true.shape[0]):
        sum_pixels = np.sum(y_true[i, ...])
        if sum_pixels == 0:
            continue
        start = i
        break

    for i in range(y_true.shape[0] - 1, -1, -1):
        sum_pixels = np.sum(y_true[i, ...])
        if sum_pixels == 0:
            continue
        stop = i
        break

    return start, stop


def interp_slice(y_true, y_pred, orientation='M'):
    dice_losses = []
    start, stop = find_start_and_end_slice(y_true)

    y_true = np.copy(y_true)
    y_pred = np.copy(y_pred)

    assert y_true.shape == y_pred.shape
    num_slices = y_true.shape[0]

    if orientation not in ['M', 'L']:
        raise ValueError('orientation must be \'M\' or \'L\'')

    if orientation is 'L':
        y_true = y_true[::-1, ...]
        y_pred = y_pred[::-1, ...]

    for i in range(num_slices):
        y_true_curr = y_true[i, ...]
        y_pred_curr = y_pred[i, ...]
        dice_losses.append(dice_score_coefficient(y_true_curr, y_pred_curr))

    dice_losses = np.asarray(dice_losses)

    xt = (np.asarray(list(range(num_slices))) - start) / (stop - start) * 100.0
    yt = dice_losses

    # interpolate only between 0 and 100%
    xp = (np.asarray(list(range(start, stop + 1))) - start) / (stop - start) * 100.0
    yp = dice_losses[start:stop + 1]

    xs = np.linspace(0, 100, 1001)
    ys = np.interp(xs, xp, yp)

    return xs, ys, xt, yt


def test_model(config, save_file=0, save_h5_data=SAVE_H5_DATA, voxel_spacing=None):
    """
    Test model
    :param config: a Config object
    :param save_file: save data (default = 0)
    """
    test_gen = get_generator(config)

    test_set_md = dict()
    for k in TEST_SET_MD.keys():
        test_set_md[k] = ScanMetadata(TEST_SET_MD[k])

    # Load config data
    test_path = config.TEST_PATH
    test_result_path = config.TEST_RESULT_PATH
    test_batch_size = config.TEST_BATCH_SIZE

    K.set_image_data_format('channels_last')

    # Load weights into Deeplabv3 model
    model = get_model(config)
    #plot_model(model, os.path.join(config.TEST_RESULT_PATH, 'model.png'), show_shapes=True)
    model.load_weights(config.TEST_WEIGHT_PATH)

    img_cnt = 1

    start = time.time()

    # Read the files that will be segmented
    test_gen.summary()
    logger.info('Save path: %s' % (test_result_path))

    # test_gen = img_generator_oai_test(test_path, test_batch_size, config)

    pids_str = ''

    x_interp = []
    y_interp = []
    x_total = []
    y_total = []

    # TODO (arjundd): remove when tissue names added to config
    class_names = mri_utils.get_tissue_name(config.TISSUES)

    metrics_manager = MetricsManager(class_names=class_names)
    seg_metrics_processor = metrics_manager.seg_metrics_processor

    # image writer
    mc_overlay = MultiClassOverlay(config.get_num_classes())

    # # Iterature through the files to be segmented
    for x_test, y_test, recon, fname, seg_time in test_gen.img_generator_test(model):
        # Perform the actual segmentation using pre-loaded model
        # Threshold at 0.5
        recon_o = np.asarray(recon)

        if config.LOSS[1] == 'sigmoid':
            # sigmoid activation function used
            labels = (recon > 0.5).astype(np.float32)
        else:
            # else, softmax used
            labels = np.zeros(recon.shape)
            l_argmax = np.argmax(recon, axis=-1)
            for c in range(labels.shape[-1]):
                labels[l_argmax == c, c] = 1
            labels = labels.astype(np.float32)

        # background is always excluded from analysis
        if config.INCLUDE_BACKGROUND:
            y_test = y_test[..., 1:]
            recon = recon[..., 1:]
            labels = labels[..., 1:]
            if y_test.ndim == 3:
                y_test = y_test[..., np.newaxis]
                recon = recon[..., np.newaxis]
                labels = labels[..., np.newaxis]

        num_slices = x_test.shape[0]
        
        if not voxel_spacing:
            voxel_spacing = get_voxel_spacing(num_slices)
        else:
            if not isinstance(voxel_spacing, tuple) or len(voxel_spacing) != 3:
                raise ValueError('voxel_spacing %s not supported.' % str(voxel_spacing))
        
        summary = metrics_manager.analyze(fname, np.transpose(y_test, axes=[1, 2, 0, 3]),
                                          np.transpose(labels, axes=[1, 2, 0, 3]),
                                          voxel_spacing=voxel_spacing,
                                          runtime=seg_time)

        logger.info_str = 'Scan #%03d (name = %s, %d slices) = %s' % (img_cnt, fname, num_slices, summary)
        pids_str = pids_str + logger.info_str + '\n'
        logger.info(logger.info_str)

        if fname in test_set_md.keys():
            slice_dir = test_set_md[fname].slice_dir

            # interpolate region of interest
            xs, ys, xt, yt = interp_slice(y_test, labels, orientation=slice_dir)
            x_interp.append(xs)
            y_interp.append(ys)
            x_total.append(xt)
            y_total.append(yt)

        if save_file == 1:
            if save_h5_data:
                save_name = '%s/%s.pred' % (test_result_path, fname)
                with h5py.File(save_name, 'w') as h5f:
                    h5f.create_dataset('recon', data=recon)
                    h5f.create_dataset('gt', data=y_test)

            # in case of 2.5D, we want to only select center slice
            x_write = x_test[..., x_test.shape[-1] // 2]

            # Save mask overlap
            # TODO (arjundd): fix writing masks to files
            #x_write_o = np.transpose(x_write, (1, 2, 0))
            #recon_oo = np.transpose(recon_o, (1, 2, 0, 3))
            #mc_overlay.im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write_o, recon_oo)
            #ovlps = im_utils.write_ovlp_masks(os.path.join(test_result_path, 'ovlp', fname), y_test[...,0], labels[...,0])
            #im_utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
            #im_utils.write_mask(os.path.join(test_result_path, 'labels', fname), labels)
            #im_utils.write_prob_map(os.path.join(test_result_path, 'prob_map', fname), recon)
            #im_utils.write_im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_write, ovlps)
            #im_utils.write_sep_im_overlay(os.path.join(test_result_path, 'im_ovlp_sep', fname), x_write,
            #                               np.squeeze(y_test), np.squeeze(labels))

        img_cnt += 1

    end = time.time()

    stats_string = get_stats_string(metrics_manager, end - start)
    # Log some summary statistics
    logger.info('--' * 20)
    logger.info(stats_string)
    logger.info('--' * 20)

    test_results_summary_path = os.path.join(test_result_path, 'results.txt')
    # Write details to test file
    with open(test_results_summary_path, 'w+') as f:
        f.write('Results generated on %s\n' % strftime('%X %x %Z'))
        f.write('Weights Loaded: %s\n' % os.path.basename(config.TEST_WEIGHT_PATH))
        f.write('Voxel Spacing: %s\n' % str(voxel_spacing))
        f.write('--' * 20)
        f.write('\n')
        f.write(pids_str)
        f.write('--' * 20)
        f.write('\n')
        f.write(stats_string)

    # os.chmod(test_results_summary_path, S_IREAD | S_IRGRP | S_IROTH)

    # Save metrics in dat format using pickle
    results_dat = os.path.join(test_result_path, 'metrics.dat')
    io_utils.save_pik(metrics_manager.data, results_dat)

    x_interp = np.asarray(x_interp)
    y_interp = np.asarray(y_interp)
    x_total = np.asarray(x_total)
    y_total = np.asarray(y_total)

    sio.savemat(os.path.join(test_result_path, 'total_interp_data.mat'), {'xs': x_interp,
                                                                          'ys': y_interp,
                                                                          'xt': x_total,
                                                                          'yt': y_total})


def get_stats_string(mw: MetricsManager, testing_time):
    """
    Return string detailing statistics
    :param mw:
    :param skipped_count:
    :param testing_time:
    :return:
    """
    seg_metrics_processor = mw.seg_metrics_processor
    inference_runtimes = np.asarray(mw.runtimes)

    s = '============Overall Summary============\n'
    s += 'Time elapsed: %0.1f seconds.\n' % testing_time
    s += '%s\n' % seg_metrics_processor.summary()
    s += 'Inference time (Mean +/- Std. Dev.): %0.2f +/- %0.2f seconds.\n' % (np.mean(inference_runtimes),
                                                                              np.std(inference_runtimes))
    return s


def check_results_file(base_path):
    if (base_path is None) or (not os.path.isdir(base_path)) or (base_path == ''):
        return []

    results_filepath = os.path.join(base_path, 'results.txt')

    results_paths = []
    if os.path.isfile(results_filepath):
        results_paths.append(results_filepath)

    files = os.listdir(base_path)
    for file in files:
        possible_dir = os.path.join(base_path, file)
        if os.path.isdir(possible_dir):
            subdir_results_files = check_results_file(possible_dir)
            results_paths.extend(subdir_results_files)

    return results_paths


def get_valid_subdirs(base_path, no_results=True):
    """
    Return subdirectories that have data to be tested
    :param base_path: root folder to search
    :param no_results: only select folders that don't have results
    :return: list of paths (strings)
    """
    if (base_path is None) or (not os.path.isdir(base_path)) or (base_path == []):
        return []

    dir_base_name = os.path.basename(base_path)
    try:
        d = strptime(dir_base_name, '%Y-%m-%d-%H-%M-%S')
        if d < DATE_THRESHOLD:
            return []
    except ValueError:
        is_date_directory = False

    subdirs = []
    config_path = os.path.join(base_path, 'config.ini')
    pik_data_path = os.path.join(base_path, 'pik_data.dat')
    test_results_dirpath = os.path.join(base_path, 'test_results')
    results_file_exists = len(check_results_file(test_results_dirpath)) > 0

    # 1. Check if you are a valid subdirectory - must contain a pik data path
    if os.path.isfile(config_path) and os.path.isfile(pik_data_path):
        if (no_results and (not results_file_exists)) or ((not no_results)):
            subdirs.append(base_path)

    files = os.listdir(base_path)
    # 2. Recursively search through other subdirectories
    for file in files:
        possible_dir = os.path.join(base_path, file)
        if os.path.isdir(possible_dir):
            rec_subdirs = get_valid_subdirs(possible_dir, no_results)
            subdirs.extend(rec_subdirs)

    return subdirs


def batch_test(base_folder, config_name, vals_dicts=[None], overwrite=False):
    # get list of directories to get info from
    subdirs = get_valid_subdirs(base_folder, not overwrite)

    for subdir in subdirs:
        logger.info(subdir)

    logger.info('')
    for subdir in subdirs:
        for vals_dict in vals_dicts:
            config = get_config(config_name)
            test_dir(subdir, config, vals_dict=vals_dict)


def find_best_test_dir(base_folder):
    subdirs = get_valid_subdirs(base_folder, no_results=False)
    max_dsc_details = (0, '')

    for subdir in subdirs:
        base_results = os.path.join(subdir, 'test_results')
        results_files = check_results_file(base_results)
        assert not ((results_files is None) or (len(results_files) == 0)), "Checking results file failed - %s" % subdir
        for results_file in results_files:
            mean = utils.parse_results_file(results_file)
            potential_data = (mean, results_file)
            logger.info(potential_data)
            if mean > max_dsc_details[0]:
                max_dsc_details = potential_data
    logger.info('\nMAX')
    logger.info(max_dsc_details)


def test_dir(dirpath, config=None, vals_dict=None, best_weight_path=None, save_h5_data=False, voxel_spacing=None):
    """
    Run testing experiment
    By default, save all data
    :param dirpath: path to directory storing model config
    :param config: a Config object
    :param vals_dict: vals_dict: a dictionary of config parameters to change (default = None)
                      e.g. {'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}
    :param best_weight_path: path to best weights (default = None)
                                if None, automatically search dirpath for the best weight path
    """
    # Create config, if not provided.
    config_filepath = os.path.join(dirpath, 'config.ini')
    if not config:
        config = MCONFIG.get_config(MCONFIG.get_cp_save_tag(config_filepath),
                                    create_dirs=False)

    # Initialize logger.
    setup_logger(config.CP_SAVE_PATH)
    logger.info('OUTPUT_DIR: %s' % config.CP_SAVE_PATH)

    # Get best weight path
    if best_weight_path is None:
        best_weight_path = dl_utils.get_weights(dirpath)
    logger.info('Best weights: %s' % best_weight_path)

    logger.info('Config: %s' % config_filepath)
    config.load_config(config_filepath)
    config.TEST_WEIGHT_PATH = best_weight_path

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.change_to_test()

    test_model(config, save_file=1, save_h5_data=save_h5_data, voxel_spacing=voxel_spacing)

    K.clear_session()


ARCHITECTURE_PATHS_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit/%s'
DATA_LIMIT_PATHS_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_seg_networks/data_limit', '%03d', '%s')
AUGMENTATION_PATH_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_seg_networks/augment_limited', '%s')
LOSS_PATH_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_seg_networks/loss_limit', '%s')
VOLUME_PATH_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_seg_networks/volume_limited', '%s')
BEST_NETWORK_PATHS_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/best_network/%s'
EXP_KEY = 'exp'
BATCH_TEST_KEY = 'batch'
SUPPORTED_ARCHITECTURES = ['unet_2d', 'deeplabv3_2d', 'segnet_2d', 'res_unet']
ARCHITECTURE_KEY = 'architecture'
OVERWRITE_KEY = 'ov'

OS_KEY = 'OS'
DIL_RATES_KEY = 'DIL_RATES'


def get_config(name):
    configs = [DeeplabV3Config(create_dirs=False), UNetConfig(create_dirs=False), SegnetConfig(create_dirs=False),
               UNet2_5DConfig(create_dirs=False), ResidualUNet(create_dirs=False)]

    for config in configs:
        if config.CP_SAVE_TAG == name:
            return config

    raise ValueError('config %s not found' % name)


def init_deeplab_parser(parser):
    parser.add_argument('-%s' % OS_KEY, nargs='?', default=None, choices=[8, 16])
    parser.add_argument('-%s' % DIL_RATES_KEY, nargs='?', default=None, type=tuple)


def handle_deeplab(vargin):
    vals_dict = dict()
    if vargin[OS_KEY]:
        vals_dict[OS_KEY] = vargin[OS_KEY]

    if vargin[DIL_RATES_KEY]:
        dil_rates = vargin[DIL_RATES_KEY]
        if type(dil_rates) is not tuple or len(dil_rates) != 3:
            raise ValueError('Dilation rates must be a tuple of 3 integers')

        vals_dict[DIL_RATES_KEY] = vargin[DIL_RATES_KEY]

    return vals_dict


def add_base_architecture_parser(architecture_parser, supported_architectures=SUPPORTED_ARCHITECTURES, add_vals=[]):
    for architecture in supported_architectures:
        parser = architecture_parser.add_parser(architecture, help='use %s' % architecture)
        parser.add_argument('-%s' % BATCH_TEST_KEY, action='store_const', default=False, const=True,
                            help='batch test directory')
        parser.add_argument('-%s' % OVERWRITE_KEY, action='store_const', default=False, const=True,
                            help='overwrite current data')

        parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                            help='gpu id to use')
        parser.add_argument('-c', '--cpu', metavar='c', action='store_const', default=False, const=True)
        parser.add_argument('-date', nargs='?')
        parser.add_argument('-batch_size', default=72, type=int, nargs='?')
        parser.add_argument('-save_h5_data', action='store_const', default=False, const=True,
                            help='save ground truth and prediction data in h5 format')

        for v in add_vals:
            parser.add_argument('-%s' % v, nargs='?', type=str, default=None)

        if architecture == 'deeplabv3_2d':
            init_deeplab_parser(parser)


def init_architecture_parser(input_subparser):
    subparser = input_subparser.add_parser('arch', help='test architecture experiment')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_architecture_exp)


def init_data_limit_parser(input_subparser):
    subparser = input_subparser.add_parser('data', help='test data limitation experiment')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_data_limit_exp)


def handle_architecture_exp(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = vargin['batch_size']

    architecture_folder_path = ARCHITECTURE_PATHS_PREFIX % config_name
    vals_dict = {'TEST_BATCH_SIZE': test_batch_size}

    if config_name == 'deeplabv3_2d':
        vals_dict.update(handle_deeplab(vargin))

    if do_batch_test:
        batch_test(architecture_folder_path, config_name, [vals_dict], overwrite=overwrite_data)
        return

    if date is None:
        raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

    fullpath = os.path.join(architecture_folder_path, date)
    if not os.path.isdir(fullpath):
        raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

    test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


def handle_data_limit_exp(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = vargin['batch_size']

    for count in [5, 15, 30, 60]:
        architecture_folder_path = DATA_LIMIT_PATHS_PREFIX % (count, config_name)

        vals_dict = {'TEST_BATCH_SIZE': test_batch_size}

        if config_name == 'deeplabv3_2d':
            vals_dict.update(handle_deeplab(vargin))

        if do_batch_test:
            batch_test(architecture_folder_path, config_name, [vals_dict], overwrite=overwrite_data)
            continue

        if date is None:
            raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

        fullpath = os.path.join(architecture_folder_path, date)
        if not os.path.isdir(fullpath):
            raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

        test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


def handle_augment_limit_exp(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = vargin['batch_size']

    augmentation_folder_path = AUGMENTATION_PATH_PREFIX % config_name
    vals_dict = {'TEST_BATCH_SIZE': test_batch_size}

    if config_name == 'deeplabv3_2d':
        vals_dict.update(handle_deeplab(vargin))

    if do_batch_test:
        batch_test(augmentation_folder_path, config_name, [vals_dict], overwrite=overwrite_data)
        return

    if date is None:
        raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

    fullpath = os.path.join(augmentation_folder_path, date)
    if not os.path.isdir(fullpath):
        raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

    test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


def init_augment_limit_parser(input_subparser):
    subparser = input_subparser.add_parser('aug', help='test augmentation experiment')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_augment_limit_exp)


def init_loss_limit_parser(input_subparser):
    subparser = input_subparser.add_parser('loss', help='test loss experiment (DSC vs weighted CE)')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_loss_limit_exp)


def handle_loss_limit_exp(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = vargin['batch_size']

    loss_folder_path = LOSS_PATH_PREFIX % config_name
    vals_dict = {'TEST_BATCH_SIZE': test_batch_size}

    if config_name == 'deeplabv3_2d':
        vals_dict.update(handle_deeplab(vargin))

    if do_batch_test:
        batch_test(loss_folder_path, config_name, [vals_dict], overwrite=overwrite_data)
        return

    if date is None:
        raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

    fullpath = os.path.join(loss_folder_path, date)
    if not os.path.isdir(fullpath):
        raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

    test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


def init_volume_limit_parser(input_subparser):
    subparser = input_subparser.add_parser('vol', help='test volume experiment (2.5D/3D)')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser, ['unet_2_5d'])

    subparser.set_defaults(func=handle_volume_limit_exp)


def handle_volume_limit_exp(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = vargin['batch_size']

    loss_folder_path = VOLUME_PATH_PREFIX % config_name
    vals_dict = {'TEST_BATCH_SIZE': test_batch_size}

    if do_batch_test:
        batch_test(loss_folder_path, config_name, [vals_dict], overwrite=overwrite_data)
        return

    if date is None:
        raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

    fullpath = os.path.join(loss_folder_path, date)
    if not os.path.isdir(fullpath):
        raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

    test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


def init_fcn_test_parser(input_subparser):
    subparser = input_subparser.add_parser('fcn',
                                           help='test fcn experiment - how well does the network generalize to non-preprocessed data')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser, add_vals=['fp'])

    subparser.set_defaults(func=handle_fcn_test_parser)


def handle_fcn_test_parser(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = 80

    folder_path = vargin['fp']
    if folder_path is None or not os.path.isdir(folder_path):
        raise ValueError('fp must be specified')

    base_path = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/test_2d_%s'
    test_folders = [('midcrop1', (320, 320, 1)), ('midcrop2', (352, 352, 1)), ('nocrop', (384, 384, 1))]

    for test_folder, img_size in test_folders:
        test_path = base_path % test_folder
        test_results_folder_name = 'test_results_%s' % test_folder

        vals_dict = {'TEST_BATCH_SIZE': test_batch_size,
                     'TEST_PATH': test_path,
                     'TEST_RESULTS_FOLDER_NAME': test_results_folder_name,
                     'IMG_SIZE': img_size}

        if config_name == 'deeplabv3_2d':
            vals_dict.update(handle_deeplab(vargin))

        if do_batch_test:
            batch_test(folder_path, config_name, [vals_dict], overwrite=overwrite_data)
            continue

        if date is None:
            raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

        fullpath = os.path.join(folder_path, config_name, date)
        if not os.path.isdir(fullpath):
            raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

        test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


def init_best_network_test_parser(input_subparser):
    subparser = input_subparser.add_parser('best', help='test best trained experiment')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_best_network_test_exp)


def handle_best_network_test_exp(vargin):
    config_name = vargin[ARCHITECTURE_KEY]
    do_batch_test = vargin[BATCH_TEST_KEY]
    overwrite_data = vargin[OVERWRITE_KEY]
    date = vargin['date']
    test_batch_size = vargin['batch_size']

    architecture_folder_path = BEST_NETWORK_PATHS_PREFIX % config_name
    vals_dict = {'TEST_BATCH_SIZE': test_batch_size}

    if config_name == 'deeplabv3_2d':
        vals_dict.update(handle_deeplab(vargin))

    if do_batch_test:
        batch_test(architecture_folder_path, config_name, [vals_dict], overwrite=overwrite_data)
        return

    if date is None:
        raise ValueError('Must specify either \'date\' or \'%s\'' % (BATCH_TEST_KEY))

    fullpath = os.path.join(architecture_folder_path, date)
    if not os.path.isdir(fullpath):
        raise NotADirectoryError('%s does not exist. Make sure date is correct' % fullpath)

    test_dir(fullpath, get_config(config_name), vals_dict=vals_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on OAI dataset')

    subparsers = parser.add_subparsers(help='experiment to run', dest=EXP_KEY)
    init_architecture_parser(subparsers)
    init_data_limit_parser(subparsers)
    init_augment_limit_parser(subparsers)
    init_loss_limit_parser(subparsers)
    init_volume_limit_parser(subparsers)
    init_fcn_test_parser(subparsers)
    init_best_network_test_parser(subparsers)

    args = parser.parse_args()
    gpu = args.gpu
    cpu = args.cpu

    if not cpu:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    vargin = vars(args)
    SAVE_H5_DATA = vargin['save_h5_data']
    args.func(vargin)
