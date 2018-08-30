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

from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig, UNetConfig
import utils


def test_model(config, save_file=0):

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
        test_gen = img_generator_oai_test(test_path, test_batch_size, img_size, config.TISSUES, tag=config.TAG)
    else:
        test_gen = img_generator_test(test_path, test_batch_size, img_size, config.TAG, config.TISSUES, shuffle_epoch=False)

    pids_str = ''

    # # Iterature through the files to be segmented
    for x_test, y_test, fname, num_slices in test_gen:

        # Perform the actual segmentation using pre-loaded model
        # Threshold at 0.5
        recon = model.predict(x_test, batch_size = test_batch_size)
        labels = (recon > 0.5).astype(np.float32)
       
        
        # Calculate real time dice coeff for analysis
        dl = dice_loss_test(labels,y_test)

        print_str = 'Dice score for image #%d (name = %s, %d slices) = %0.3f' % (img_cnt, fname, num_slices, np.mean(dl))
        pids_str = pids_str + print_str + '\n'
        print(print_str)

        if (save_file == 1):
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


def get_stats_string(dice_losses, skipped_count, testing_time):
    s = 'Overall Summary:\n'
    s += '%d Skipped\n' % skipped_count
    s += 'Mean= %0.4f Std = %0.3f\n' % (np.mean(dice_losses), np.std(dice_losses))
    s += 'Median = %0.4f\n' % np.median(dice_losses)
    s += 'Time required = %0.1f seconds.\n'% testing_time
    return s


def get_valid_subdirs(base_path):
    if base_path is None:
        return []

    # Find all subdirectories
    # subdirectory is valid if it contains a 'config.txt' file
    files = os.listdir(base_path)
    subdirs = []
    for file in files:
        if (os.path.isdir(os.path.join(base_path, file)) and os.path.isfile(os.path.join(base_path, file, 'config.ini'))):
            subdir = os.path.join(base_path, file)
            subdirs.append(subdir)

            # recursively search directories for any folders that have similar setup
            rec_subdirs = get_valid_subdirs(subdir)
            subdirs.append(rec_subdirs)

    return subdirs


# def batch_test(base_folder):
#     # get list of directories to get info from
#     subdirs = get_valid_subdirs(base_folder)
#
#     for subdir in subdirs:
#         test_dir(subdir)


def test_dir(dirpath, config, vals_dict=None, best_weight_path=None):

    # Get best weight path
    if best_weight_path is None:
        best_weight_path = utils.get_weights(dirpath)
    print('Best weight path: %s' % best_weight_path)

    config.load_config(os.path.join(dirpath, 'config.ini'))
    config.TEST_WEIGHT_PATH = best_weight_path

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.change_to_test()

    test_model(config, save_file=1)

    K.clear_session()


local_testing_test_path = '../sample_data/test_data'
local_test_results_path = '../sample_data/results'

DEEPLAB_TEST_PATHS_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d'
DEEPLAB_TEST_PATHS = ['2018-08-26-20-01-32', # OS=16, DIL_RATES=(6, 12, 18)
                      '2018-08-27-02-49-06', # OS=16, DIL_RATES=(1, 9, 18)
                      '2018-08-27-15-48-56', # OS=16, DIL_RATES=(3, 6, 9)
                      ]
DEEPLAB_DIL_RATES = [ [(6, 12, 18), (3, 6, 9), (2, 4, 6), (1, 2, 3), (12, 24, 36)],
                      [(1, 9, 18), (1, 3, 6), (1, 2, 4), (1, 1, 2)],
                      [(3, 6, 9), (6, 12, 18), (2, 4, 6), (1, 2, 3), (12, 24, 36)],
                    ]

DATA_LIMIT_PATHS_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_data_limit/oai_data', '%03d', 'unet_2d')
DATA_LIMIT_NUM_DATE_DICT = {5:'2018-08-26-20-19-31',
                            15:'2018-08-27-03-43-46',
                            30:'2018-08-27-11-18-07',
                            60:'2018-08-27-18-29-19'}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OAI dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
    args = parser.parse_args()
    gpu = args.gpu

    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
                
    # Test data limit
    #for num_subjects in DATA_LIMIT_NUM_DATE_DICT.keys():
     #   date_str = DATA_LIMIT_NUM_DATE_DICT[num_subjects]
      #  filepath = os.path.join(DATA_LIMIT_PATHS_PREFIX % num_subjects, date_str)
       # config = UNetConfig(create_dirs=False)
       # test_dir(filepath, config)

    config = DeeplabV3Config(create_dirs=False)
    test_dir(os.path.join(DEEPLAB_TEST_PATHS_PREFIX, '2018-08-29-05-39-09'), config, {'OS':8, 'DIL_RATES':(12, 24, 36), 'TEST_BATCH_SIZE':9})
