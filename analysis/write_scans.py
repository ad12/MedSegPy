import sys
sys.path.insert(0, '../')

import os
import SimpleITK as sitk
import numpy as np

from im_generator import img_generator_oai_test
from config import UNetConfig
import utils

test_path = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'

def write_tiff(x, filepath):
    print('Saving %s' % filepath)

    x = np.squeeze(x)

    sitk.WriteImage(x, filepath)


SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data_test'
utils.check_dir(SAVE_PATH)

if __name__ == '__main__':
    config = UNetConfig(create_dirs=False)

    for x, y, pid, num_slices in img_generator_oai_test(config.TEST_PATH, config.TEST_BATCH_SIZE, config):
        write_tiff(x, os.path.join(SAVE_PATH, pid + '.tiff'))
