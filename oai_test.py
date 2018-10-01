# Author: Zhongnan Fang, zhongnanf@gmail.com, 2017 July
# Modified: Akshay Chaudhari, akshaysc@stanford.edu 2017 August
#           Arjun Desai, arjundd@stanford.edu, 2018 June

from __future__ import print_function, division

import argparse
import numpy as np
import h5py
import time
import os

from keras import backend as K

from im_generator import img_generator_test, calc_generator_info, img_generator_oai_test
from losses import dice_loss_test
from models import get_model

import config as MCONFIG
from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig, UNetConfig
import utils
import scipy.ndimage as sni
import scipy.io as sio

def find_start_and_end_slice(y_true):
    for i in range(y_true.shape[0]):
        sum_pixels = np.sum(y_true[i, ...])
        if sum_pixels == 0:
            continue
        start = i
        break

    for i in range(y_true.shape[0]-1, -1, -1):
        sum_pixels = np.sum(y_true[i, ...])
        if sum_pixels == 0:
            continue
        stop = i
        break

    return start, stop


def interp_slice(y_true, y_pred):
    dice_losses = []
    start, stop = find_start_and_end_slice(y_true)
    for i in range(start, stop+1):
        dice_losses.append(dice_loss_test(y_true, y_pred))

    dice_losses = sni.zoom(dice_losses, 1001)
    xs = np.linspace(0, 100, 1001)

    return xs, dice_losses


def test_model(config, save_file=0):
    """
    Test model
    :param config: a Config object
    :param save_file: save data (default = 0)
    """

    # Load config data
    test_path = config.TEST_PATH
    test_result_path = config.TEST_RESULT_PATH
    test_batch_size = config.TEST_BATCH_SIZE

    K.set_image_data_format('channels_last')

    img_size = config.IMG_SIZE

    # Load weights into Deeplabv3 model
    model = get_model(config)
    model.load_weights(config.TEST_WEIGHT_PATH, by_name=True)

    img_cnt = 0
    dice_losses = np.array([])
    start = time.time()
    skipped_count = 0

    # Read the files that will be segmented
    test_files, ntest = calc_generator_info(test_path, test_batch_size)
    print('INFO: Test size: %d, batch size: %d, # subjects: %d' % (len(test_files), test_batch_size, ntest))
    print('Save path: %s' % (test_result_path))

    if (config.VERSION > 1):
        test_gen = img_generator_oai_test(test_path, test_batch_size, config)
    else:
        test_gen = img_generator_test(test_path, test_batch_size, img_size, config.TAG, config.TISSUES, shuffle_epoch=False)

    pids_str = ''

    interp_dice_losses = []

    # # Iterature through the files to be segmented
    for x_test, y_test, fname, num_slices in test_gen:

        # Perform the actual segmentation using pre-loaded model
        # Threshold at 0.5
        recon = model.predict(x_test, batch_size = test_batch_size)
        if (config.INCLUDE_BACKGROUND):
            y_test = y_test[..., 1]
            recon = recon[..., 1]
            y_test = y_test[..., np.newaxis]
            recon = recon[..., np.newaxis]
        labels = (recon > 0.5).astype(np.float32)

        # Calculate real time dice coeff for analysis
        dl = dice_loss_test(y_test, labels)
        dice_losses = np.append(dice_losses, dl)
        print_str = 'Dice score for image #%d (name = %s, %d slices) = %0.3f' % (img_cnt, fname, num_slices, np.mean(dl))
        pids_str = pids_str + print_str + '\n'
        print(print_str)

        # interpolate region of interest
        xs, interp = interp_slice(y_test, labels)
        interp_dice_losses.append(interp)

        if save_file == 1:
            save_name = '%s/%s_recon.pred' %(test_result_path,fname)
            with h5py.File(save_name,'w') as h5f:
                h5f.create_dataset('recon',data=recon)
            
            # Save mask overlap
            ovlps = utils.write_ovlp_masks(os.path.join(test_result_path, 'ovlp', fname), y_test, labels)
            utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
            utils.write_prob_map(os.path.join(test_result_path, 'prob_map', fname), recon)
            utils.write_im_overlay(os.path.join(test_result_path, 'im_ovlp', fname), x_test, ovlps)
        
        img_cnt += 1
        
        if img_cnt == ntest:
            break

    end = time.time()

    stats_string = get_stats_string(dice_losses, skipped_count, end-start)
    # Print some summary statistics
    print('--'*20)
    print(stats_string)
    print('--'*20)

    # Write details to test file
    with open(os.path.join(test_result_path, 'results.txt'), 'w+') as f:
        f.write('--'*20)
        f.write('\n')
        f.write(pids_str)
        f.write('--'*20)
        f.write('\n')
        f.write(stats_string)

    sio.savemat('total_interp_data.mat', {'xs':xs, 'ys':np.asarray(interp_dice_losses)})


def get_stats_string(dice_losses, skipped_count, testing_time):
    """
    Return string detailing statistics
    :param dice_losses: list of dice losses per exam
    :param skipped_count: number of exams skipped
    :param testing_time: time to run tests for all exams
    :return: a string
    """
    s = 'Overall Summary:\n'
    s += '%d Skipped\n' % skipped_count
    s += 'Mean +/- Std = %0.4f +/- %0.3f\n' % (np.mean(dice_losses), np.std(dice_losses))
    s += 'Median = %0.4f\n' % np.median(dice_losses)
    s += 'Time required = %0.1f seconds.\n'% testing_time
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

    subdirs = []
    config_path = os.path.join(base_path, 'config.ini')
    test_results_dirpath = os.path.join(base_path, 'test_results')
    results_file_exists = len(check_results_file(test_results_dirpath)) > 0

    # 1. Check if you are a valid subdirectory
    if os.path.isfile(config_path):
        if (no_results and (not results_file_exists)) or ((not no_results) and results_file_exists):
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
        print(subdir)

    print('')
    for subdir in subdirs:
        for vals_dict in vals_dicts:
            config = get_config(config_name)

            try:
                test_dir(subdir, config, vals_dict=vals_dict)
            except:
                print('Failed for %s\n' % subdir)
                break


def find_best_test_dir(base_folder):
    subdirs = get_valid_subdirs(base_folder, no_results=False)
    max_dsc_details = (0, '')

    for subdir in subdirs:
        base_results = os.path.join(subdir, 'test_results')
        results_files = check_results_file(base_results)
        assert not((results_files is None) or (len(results_files) == 0)), "Checking results file failed - %s" % subdir
        for results_file in results_files:
            mean = utils.parse_results_file(results_file)
            potential_data = (mean, results_file)
            print(potential_data)
            if mean > max_dsc_details[0]:
                max_dsc_details = potential_data
    print('\nMAX')
    print(max_dsc_details)


def test_dir(dirpath, config, vals_dict=None, best_weight_path=None):
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
    # Get best weight path
    if best_weight_path is None:
        best_weight_path = utils.get_weights(dirpath)
    print('Best weight path: %s' % best_weight_path)
    if (type(config) is not UNetConfig):
        config.load_config(os.path.join(dirpath, 'config.ini'))
    else:
        config.CP_SAVE_PATH = dirpath

    config.TEST_WEIGHT_PATH = best_weight_path

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.change_to_test()

    test_model(config, save_file=1)

    K.clear_session()


ARCHITECTURE_PATHS_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/%s'
DATA_LIMIT_PATHS_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_data_limit/oai_data', '%03d', '%s')

EXP_KEY='exp'
BATCH_TEST_KEY = 'batch'
SUPPORTED_ARCHITECTURES = ['unet_2d', 'deeplabv3_2d', 'segnet_2d']
ARCHITECTURE_KEY = 'architecture'
OVERWRITE_KEY = 'ov'

OS_KEY = 'OS'
DIL_RATES_KEY='DIL_RATES'


def get_config(name):
    configs = [DeeplabV3Config(create_dirs=False), UNetConfig(create_dirs=False), SegnetConfig(create_dirs=False)]

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


def add_base_architecture_parser(architecture_parser):
    for architecture in SUPPORTED_ARCHITECTURES:
        parser = architecture_parser.add_parser(architecture, help='use %s' % architecture)
        parser.add_argument('-%s' % BATCH_TEST_KEY, action='store_const', default=False, const=True,
                            help='batch test directory')
        parser.add_argument('-%s' % OVERWRITE_KEY, action='store_const',default=False, const=True,
                            help='overwrite current data')

        parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                            help='gpu id to use')
        parser.add_argument('-c', '--cpu', metavar='c', action='store_const', default=False, const=True)
        parser.add_argument('-date', nargs='?')
        parser.add_argument('-batch_size', default=72, type=int, nargs='?')

        if architecture == 'deeplabv3_2d':
            init_deeplab_parser(parser)


def init_architecture_parser(input_subparser):
    subparser = input_subparser.add_parser('arch', help='test architecture experiment')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_architecture_exp)


def init_data_limit_parser(input_subparser):
    subparser = input_subparser.add_parser('dl', help='test data limitation experiment')
    architecture_parser = subparser.add_subparsers(help='architecture to use', dest=ARCHITECTURE_KEY)

    add_base_architecture_parser(architecture_parser)

    subparser.set_defaults(func=handle_data_limit_exp)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OAI dataset')

    subparsers = parser.add_subparsers(help='experiment to run', dest=EXP_KEY)
    init_architecture_parser(subparsers)
    init_data_limit_parser(subparsers)

    args = parser.parse_args()
    gpu = args.gpu
    cpu = args.cpu

    if not cpu:
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
        os.environ['CUDA_VISIBLE_DEVICES']=gpu

    vargin = vars(args)

    args.func(vargin)


