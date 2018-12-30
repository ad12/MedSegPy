import os

import utils
from im_generator import get_class_freq

TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug'
CLASS_FREQ_DAT_FOLDER = utils.check_dir('/bmrNAS/people/arjun/msk_seg_networks/class_weights')

CLASS_FREQ_DAT_WEIGHTS_AUG = os.path.join(CLASS_FREQ_DAT_FOLDER, 'class_frequencies-aug.dat')
CLASS_FREQ_DAT_WEIGHTS_NO_AUG = os.path.join(CLASS_FREQ_DAT_FOLDER, 'class_frequencies-no_aug.dat')

# Create list of pids
if __name__ == '__main__':
    freq = get_class_freq(TRAIN_PATH)
    print(freq)
    utils.save_pik(freq, CLASS_FREQ_DAT_WEIGHTS_AUG)

    freq = get_class_freq(TRAIN_PATH, augment_data=False)
    print(freq)
    utils.save_pik(freq, CLASS_FREQ_DAT_WEIGHTS_NO_AUG)
