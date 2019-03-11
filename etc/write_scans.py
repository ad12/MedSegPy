import sys

sys.path.insert(0, '../')

import os
import SimpleITK as sitk
import numpy as np

from generators.im_generator import img_generator_oai_test
from config import UNetConfig
from utils import io_utils
import pandas as pd


def write_tiff(x, filepath):
    print('Saving %s' % filepath)
    x = np.squeeze(x)
    x_img = sitk.GetImageFromArray(normalize_im(x))
    x_img = sitk.Cast(x_img, sitk.sitkFloat32)

    sitk.WriteImage(x_img, filepath)


def normalize_im(x):
    return (x - np.min(x)) / np.max(x)


SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_metadata'
io_utils.check_dir(SAVE_PATH)

if __name__ == '__main__':
    config = UNetConfig(create_dirs=False)
    pids = []
    for s_path in [config.TRAIN_PATH, config.VALID_PATH, config.TEST_PATH]:
        for x, y, pid, num_slices in img_generator_oai_test(s_path, config.TEST_BATCH_SIZE, config):
            write_tiff(x, os.path.join(SAVE_PATH, pid + '.tiff'))
            pids.append(pid)

    data = list(zip(*[iter(pids)] * 1))

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(os.path.join(SAVE_PATH, 'oai_data.xlsx'))
    df.to_excel(writer)
    writer.save()
