# Author: Zhongnan Fang, zhongnanf@gmail.com, 2017 July
# Modified: Akshay Chaudhari, akshaysc@stanford.edu 2017 August
#           Arjun Desai, arjun.desai@duke.edu, 2018 June

from __future__ import print_function, division

import os
import time

import h5py
import numpy as np
from keras import backend as K

import utils
from config import SegnetConfig
from im_generator import img_generator_test, calc_generator_info
from models import get_model
from utils.metric_utils import dice_score_coefficient


def test_model(config, save_file=1):
    # set image format to be (N, dim1, dim2, dim3, ch)

    test_path = config.TEST_PATH
    test_result_path = config.TEST_RESULT_PATH
    test_batch_size = config.TEST_BATCH_SIZE

    K.set_image_data_format('channels_last')

    img_size = dlc.IMG_SIZE

    # Load weights into Deeplabv3 model
    model = get_model(config)
    model.load_weights(config.TEST_WEIGHT_PATH, by_name=True)

    img_cnt = 0
    dice_losses = np.array([])
    start = time.time()

    # Read the files that will be segmented
    test_files, ntest = calc_generator_info(test_path, test_batch_size)
    print('INFO: Test size: %d, batch size: %d, # subjects: %d' % (len(test_files), test_batch_size, ntest))
    print('Save path: %s' % (test_result_path))

    # # Iterature through the files to be segmented
    for x_test, y_test, fname in img_generator_test(test_path, test_batch_size,
                                                    img_size, dlc.TAG, dlc.TISSUES, shuffle_epoch=False):

        # Perform the actual segmentation using pre-loaded model
        recon = model.predict(x_test, batch_size=test_batch_size)
        labels = (recon > 0.5).astype(np.float32)

        # Calculate real time dice coeff for analysis
        # TODO: Define multi-class dice loss during testing
        dl = dice_score_coefficient(labels, y_test)
        dice_losses = np.append(dice_losses, dl)
        # print(dl)

        print('Dice score for image #%d (name = %s) = %0.3f' % (img_cnt, fname, np.mean(dl)))

        if (save_file == 1):
            save_name = '%s/%s_recon.pred' % (test_result_path, fname)
            with h5py.File(save_name, 'w') as h5f:
                h5f.create_dataset('recon', data=recon)

            # Save mask overlap
            save_mask_dir = os.path.join(test_result_path, 'ovlp', fname)
            utils.write_ovlp_masks(save_mask_dir, y_test, labels)
            utils.write_mask(os.path.join(test_result_path, 'gt', fname), y_test)

        img_cnt += 1

        if img_cnt == ntest:
            break

    end = time.time()

    # Print some summary statistics
    print('--' * 20)
    print('Overall Summary:')
    print('Mean= %0.4f Std = %0.3f' % (np.mean(dice_losses), np.std(dice_losses)))
    print('Median = %0.4f' % np.median(dice_losses))
    print('Time required = %0.1f seconds.' % (end - start))
    print('--' * 20)


local_testing_test_path = '../sample_data/test_data'
local_test_results_path = '../sample_data/results'

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    # set config based on what you want to train
    config = SegnetConfig(state='testing')
    test_model(config)
