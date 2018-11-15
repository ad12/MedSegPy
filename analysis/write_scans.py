import sys

sys.path.insert(0, '../')

import os
import SimpleITK as sitk
import numpy as np

from im_generator import img_generator_oai_test
from config import UNetConfig
import utils
import pandas as pd

test_path = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'


def write_tiff(x, filepath):
    print('Saving %s' % filepath)
    x = np.squeeze(x)
    x_img = sitk.GetImageFromArray(normalize_im(x))
    x_img = sitk.Cast(x_img, sitk.sitkFloat32)

    sitk.WriteImage(x_img, filepath)


def normalize_im(x):
    return (x - np.min(x)) / np.max(x)


SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data_test'
utils.check_dir(SAVE_PATH)

if __name__ == '__main__':
    config = UNetConfig(create_dirs=False)
    pids = []
    for x, y, pid, num_slices in img_generator_oai_test(config.TEST_PATH, config.TEST_BATCH_SIZE, config):
        write_tiff(x, os.path.join(SAVE_PATH, pid + '.tiff'))
        pids.append(pid)

    data = list(zip(*[iter(pids)] * 1))

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(os.path.join(SAVE_PATH, 'oai_test_data.xlsx'))
    df.to_excel(writer)
    writer.save()
