import logging
import os
import sys
import SimpleITK as sitk
import numpy as np
import pandas as pd
import time

# Matplotlib initialization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.insert(0, '../')
from medsegpy.data.im_generator import img_generator_oai_test
from medsegpy.config import UNetConfig
from medsegpy.utils import io_utils

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))


def write_tiff(x, filepath):
    logger.info('Saving %s' % filepath)
    x = np.squeeze(x)
    x_img = sitk.GetImageFromArray(normalize_im(x))
    x_img = sitk.Cast(x_img, sitk.sitkFloat32)

    sitk.WriteImage(x_img, filepath)


def write_subplot(x, filepath):
    logger.info('Saving %s' % filepath)
    x = np.squeeze(x)
    x = normalize_im(x)

    nrows = 5
    ncols = 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15,15))
    count = 0
    for i in np.linspace(0, x.shape[0]-1, nrows*ncols):
        slice_ind = int(i)
        slice_title = 'Slice %d' % (slice_ind + 1)
        ax = axs[int(count / ncols)][count % ncols]
        ax.imshow(x[slice_ind, ...], cmap='gray')
        ax.set_title(slice_title)
        ax.axis('off')
        count += 1
    plt.savefig(filepath)
    plt.close()


def normalize_im(x):
    return (x - np.min(x)) / np.max(x)


SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_metadata'
io_utils.check_dir(SAVE_PATH)

if __name__ == '__main__':
    start_time = time.time()
    config = UNetConfig(create_dirs=False)
    pids = []
    dirpaths = [config.TRAIN_PATH, config.VALID_PATH, config.TEST_PATH]
    #dirpaths = [config.TEST_PATH]
    for s_path in dirpaths:
        for x, y, pid, num_slices in img_generator_oai_test(s_path, config.TEST_BATCH_SIZE, config):
            #write_tiff(x, os.path.join(SAVE_PATH, pid + '.tiff'))
            write_subplot(x, os.path.join(SAVE_PATH, pid + '.png'))
            pids.append(pid)

    data = list(zip(*[iter(pids)] * 1))

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(os.path.join(SAVE_PATH, 'oai_data.xlsx'))
    df.to_excel(writer)
    writer.save()

    logger.info('Time Elapsed: %0.2f seconds' % (time.time() - start_time))
