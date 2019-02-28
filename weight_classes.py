import os
from os import listdir
from os.path import splitext

import h5py
import numpy as np

from generators.im_generator import add_file
from utils import io_utils

TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug'
CLASS_FREQ_DAT_FOLDER = io_utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/class_weights')

CLASS_FREQ_DAT_WEIGHTS_AUG = os.path.join(CLASS_FREQ_DAT_FOLDER, 'class_frequencies-aug.dat')
CLASS_FREQ_DAT_WEIGHTS_NO_AUG = os.path.join(CLASS_FREQ_DAT_FOLDER, 'class_frequencies-no_aug.dat')

# Create list of pids

# TODO: support weighting more than 1 class
def get_class_freq(data_path, class_id=[0], pids=None, augment_data=True):
    if pids is not None:
        learn_files = []

    files = listdir(data_path)
    unique_filename = {}

    for file in files:
        file, _ = splitext(file)
        if add_file(file, unique_filename, pids, augment_data):
            unique_filename[file] = file

    files = list(unique_filename.keys())

    # organized as freq = [background, class]
    freqs = np.zeros([len(class_id) + 1, 1])

    count = 0
    for file in files:
        seg_path = '%s/%s.seg' % (data_path, file)
        with h5py.File(seg_path, 'r') as f:
            seg = f['data'][:].astype('float32')
            # select class of interest
            seg = seg[..., class_id]
            seg = seg.flatten()

            freqs[0] += np.sum(seg == 0)
            freqs[1] += np.sum(seg == 1)

        count += 1

        if count % 1000 == 0:
            print('%d/%d' % (count, len(files)))
    return freqs


if __name__ == '__main__':
    freq = get_class_freq(TRAIN_PATH)
    print(freq)
    io_utils.save_pik(freq, CLASS_FREQ_DAT_WEIGHTS_AUG)

    freq = get_class_freq(TRAIN_PATH, augment_data=False)
    print(freq)
    io_utils.save_pik(freq, CLASS_FREQ_DAT_WEIGHTS_NO_AUG)
