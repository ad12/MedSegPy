# Author: Zhongnan Fang, zhongnanf@gmail.com, 2017 July
# Modified: Akshay Chaudhari, akshaysc@stanford.edu 2017 August
#           Arjun Desai, arjundd@stanford.edu, 2018 June

from __future__ import print_function, division

import numpy as np
import h5py
import time
import os

from keras import backend as K

from im_generator import img_generator_test, calc_generator_info
from losses import dice_loss_test
from models import get_model

from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig, UNetConfig
import utils


def test_model(config, save_file=1):
    # set image format to be (N, dim1, dim2, dim3, ch)

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

    # # Iterature through the files to be segmented
    for x_test, y_test, fname in img_generator_test(test_path, test_batch_size,
                                                img_size, config.TAG, config.TISSUES, shuffle_epoch=False):

        # Perform the actual segmentation using pre-loaded model
        recon = model.predict(x_test, batch_size = test_batch_size)
        labels = (recon > 0.5).astype(np.float32)
       
        
        # Calculate real time dice coeff for analysis
        # TODO: Define multi-class dice loss during testing
        dl = dice_loss_test(labels,y_test)
        skipped=''
        if (dl > 0.15):
            dice_losses = np.append(dice_losses,dl)
        else:
            skipped = '- skipped'
            skipped_count += 1
        # print(dl)

        print('Dice score for image #%d (name = %s) = %0.3f %s'%(img_cnt, fname, np.mean(dl), skipped))

        if (save_file == 1):
            save_name = '%s/%s_recon.pred' %(test_result_path,fname)
            with h5py.File(save_name,'w') as h5f:
                h5f.create_dataset('recon',data=recon)
            
            # Save mask overlap
            save_mask_dir = os.path.join(test_result_path, 'ovlp', fname)
            utils.write_ovlp_masks(save_mask_dir, y_test, labels)
            utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)
        
        img_cnt += 1
        
        if img_cnt == ntest:
            break

    end = time.time()

    # Print some summary statistics
    print('--'*20)
    print('Overall Summary:')
    print('%d Skipped' % skipped_count)
    print('Mean= %0.4f Std = %0.3f' % (np.mean(dice_losses),np.std(dice_losses)))
    print('Median = %0.4f' % np.median(dice_losses))
    print('Time required = %0.1f seconds.'%(end-start))
    print('--'*20)


def get_valid_subdirs(base_path):
    if base_path is None:
        return []

    # Find all subdirectories
    # subdirectory is valid if it contains a 'config.txt' file
    files = os.listdir(base_path)
    subdirs = []
    for file in files:
        if (os.path.isdir(file) and os.path.isfile(os.path.join(base_path, file, 'config.txt'))):
            subdir = os.path.join(base_path, file)
            subdirs.append(subdir)

            # recursively search directories for any folders that have similar setup
            rec_subdirs = get_valid_subdirs(subdir)
            subdirs.append(rec_subdirs)

    return subdirs


def get_weights(base_folder):
    """
    Gets the best weights file inside the base_folder
    :param base_folder: dirpath where weights are stored
    :return: h5 file

    Assumes that only the best weights are stored, so searching for the epoch should be enough
    """
    files = os.listdir(base_folder)
    max_epoch = -1
    best_file = ''
    for file in files:
        # Ensure the file is an h5 file
        if not(os.path.isfile(file) and file.endswith('.h5')):
            continue

        # Get file with max epochs
        train_info = file.split('.')[1]
        epoch = int(train_info.split('-')[0])

        if (epoch > max_epoch):
            max_epoch = epoch
            best_file = file

    return os.path.join(base_folder, best_file)


def batch_test(base_folder):
    # get list of directories to get info from
    subdirs = get_valid_subdirs(base_folder)

    for subdir in subdirs:
        # Initialize config


        # Get best weight path
        config.TEST_WEIGHT_PATH = get_weights(subdir)


local_testing_test_path = '../sample_data/test_data'
local_test_results_path = '../sample_data/results'

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    os.environ['CUDA_VISIBLE_DEVICES']="3"

    # set config based on what you want to train
    config = UNetConfig(state='testing')
    test_model(config)
