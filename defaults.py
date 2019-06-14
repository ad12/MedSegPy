SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data'  # Put absolute path for where data should be stored
TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/'
VALID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/'
TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'


# Check if save path is initialized.
if not SAVE_PATH:
    raise ValueError('Set `SAVE_PATH` in defaults.py')

