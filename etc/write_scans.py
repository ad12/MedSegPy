import os, sys
import SimpleITK as sitk
import numpy as np
import pandas as pd

# Matplotlib initialization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


sys.path.insert(0, '../')
from generators.im_generator import img_generator_oai_test
from config import UNetConfig
from utils import io_utils


def write_tiff(x, filepath):
    print('Saving %s' % filepath)
    x = np.squeeze(x)
    x_img = sitk.GetImageFromArray(normalize_im(x))
    x_img = sitk.Cast(x_img, sitk.sitkFloat32)

    sitk.WriteImage(x_img, filepath)


def write_subplot(x, filepath):
    print('Saving %s' % filepath)
    x = np.squeeze(x)
    x = normalize_im(x)

    nrows = 5
    ncols = 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30,30))
    count = 0
    for i in np.linspace(0, x.shape[0], nrows*ncols):
        slice_ind = int(i)
        slice_title = 'Slice %d' % (slice_ind + 1)
        ax = axs[int(count / ncols)][count % ncols]
        ax.imshow(x[slice_ind, ...])
        ax.set_title(slice_title)
    plt.savefig(filepath)


def normalize_im(x):
    return (x - np.min(x)) / np.max(x)


SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_metadata'
io_utils.check_dir(SAVE_PATH)

if __name__ == '__main__':
    config = UNetConfig(create_dirs=False)
    pids = []
    for s_path in [config.TRAIN_PATH, config.VALID_PATH, config.TEST_PATH]:
        for x, y, pid, num_slices in img_generator_oai_test(s_path, config.TEST_BATCH_SIZE, config):
            #write_tiff(x, os.path.join(SAVE_PATH, pid + '.tiff'))
            write_subplot(x, os.path.join(SAVE_PATH, pid + '.png'))
            pids.append(pid)

    data = list(zip(*[iter(pids)] * 1))

    df = pd.DataFrame(data)
    writer = pd.ExcelWriter(os.path.join(SAVE_PATH, 'oai_data.xlsx'))
    df.to_excel(writer)
    writer.save()
