import os
import utils
from im_generator import get_class_freq


TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug'
CLASS_FREQ_DAT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/class_frequencies.dat'

# Create list of pids
if __name__ == '__main__':
    freq = get_class_freq(TRAIN_PATH)
    print(freq)
    utils.save_pik(freq, CLASS_FREQ_DAT_PATH)
