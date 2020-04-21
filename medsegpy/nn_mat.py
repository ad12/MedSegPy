import argparse

from medsegpy.oai_test import get_valid_subdirs

import os
import time

import numpy as np
import scipy.io as sio
from keras import backend as K

from medsegpy.utils import dl_utils
from medsegpy.utils import io_utils

from medsegpy import config as MCONFIG
from medsegpy.modeling import get_model
from medsegpy.data.im_gens import get_generator

_EXPECTED_SHAPE = (288, 288, 72)


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use. default=0')
    parser.add_argument('--cpu', action='store_const', default=False, const=True,
                        help='use cpu. will overridie `-g` gpu flag')

    parser.add_argument('--batch_size', default=72, type=int, nargs='?')
    parser.add_argument('--save_gt', action='store_true',
                        help='save ground truth')
    parser.add_argument('--tag', default=None, nargs='?', type=str,
                        help='change tag for inference')
    parser.add_argument('--img_size', default=None, nargs='?')

    parser.add_argument('-r', '--recursive', action='store_true',
                        help='recursively analyze all directories')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite existing test folders')


def create_config_dict(vargin):
    config_dict = {'TEST_BATCH_SIZE': vargin['batch_size']}
    if vargin['tag']:
        config_dict['TAG'] = vargin['tag']
        config_dict['TEST_RESULTS_FOLDER_NAME'] = 'test_results_%s' % vargin['tag']

    return config_dict


def test_dir(dirpath, config=None, vals_dict=None, best_weight_path=None, save_gt=False):
    """Run test for given dirpath

    Args:
        dirpath (str): path to directory storing model config
        config: a Config object
        vals_dict (Dict): Config parameters to change (default = None).
            e.g. ``{'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}``
        best_weight_path (str): path to best weights (default = None)
            if None, automatically search dirpath for the best weight path
    """
    # Get best weight path
    if best_weight_path is None:
        best_weight_path = dl_utils.get_weights(dirpath)
    print('Best weights: %s' % best_weight_path)

    config_filepath = os.path.join(dirpath, 'config.ini')
    if not config:
        config = MCONFIG.get_config(MCONFIG.get_model_name(config_filepath), create_dirs=False)

    print('Config: %s' % config_filepath)
    config.merge_from_file(config_filepath)
    config.TEST_WEIGHT_PATH = best_weight_path

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.change_to_test()

    test_model(config, save_gt=save_gt)

    K.clear_session()


def test_model(config, save_gt=False):
    """Run inference for model and save predicted outputs as mat file.

    To perform thickness calculations, we need to save the masks in the expected `.mat` format.
    """
    test_gen = get_generator(config)

    # Load config data
    test_result_path = config.TEST_RESULT_PATH
    K.set_image_data_format('channels_last')

    # Load weights into Deeplabv3 model
    model = get_model(config)
    model.load_weights(config.TEST_WEIGHT_PATH)

    img_cnt = 0

    start_time = time.time()

    # Read the files that will be segmented
    test_gen.summary()
    print("Save Path: {}".format(test_result_path))

    fnames = []

    # Iterate through files to be segmented.
    for x_test, y_test, recon, fname in test_gen.img_generator_test(model):
        # Perform the actual segmentation using pre-loaded model
        # Threshold at 0.5
        if config.INCLUDE_BACKGROUND:
            y_test = y_test[..., 1]
            recon = recon[..., 1]
            y_test = y_test[..., np.newaxis]
            recon = recon[..., np.newaxis]
        labels = (recon > 0.5).astype(np.float32)

        # Format masks in (Y,X,Z) format.
        labels = np.squeeze(labels)
        y_test = np.squeeze(y_test)
        labels = np.transpose(labels, (1, 2, 0))
        y_test = np.transpose(y_test, (1, 2, 0))

        fnames.append(fname)

        assert labels.ndim == 3, "Shape: {}".format(labels.shape)
        #assert labels.shape == _EXPECTED_SHAPE, "Shape: {}".format(labels.shape)
        save_name = os.path.join(test_result_path, "pred_mat", "{}.mat".format(fname))
        io_utils.check_dir(os.path.dirname(save_name))
        sio.savemat(save_name, {"fcMask": labels})

        # Save ground truth masks.
        if save_gt:
            assert y_test.ndim == 3, "Shape: {}".format(y_test.shape)
            #assert y_test.shape == _EXPECTED_SHAPE, "Shape: {}".format(y_test.shape)
            save_name = os.path.join(test_result_path, "gt_mat", "{}.mat".format(fname))
            io_utils.check_dir(os.path.dirname(save_name))
            sio.savemat(save_name, {"fcMask": y_test})

        img_cnt += 1
        print("Finished {}: {}".format(img_cnt, fname))

    print("Time Elapsed: {}".format(time.time() - start_time))


if __name__ == '__main__':
    base_parser = argparse.ArgumentParser(description='Save output to mat files')
    add_testing_arguments(base_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    config_filepath = vargin['dirpath'][0]
    if not os.path.isdir(config_filepath):
        raise NotADirectoryError('Directory %s does not exist.' % config_filepath)

    gpu = args.gpu
    cpu = args.cpu
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if not cpu:
        print('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    recursive = args.recursive
    overwrite = args.force

    test_dirpaths = [config_filepath]
    if recursive:
        test_dirpaths = get_valid_subdirs(config_filepath, not overwrite)

    for dp in test_dirpaths:
        test_dir(dp, vals_dict=create_config_dict(vargin), save_gt=vargin['save_gt'])
