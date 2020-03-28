SAVE_PATH = '/bmrNAS/people/arjun/results/abCT'  # Put absolute path for where data should be stored
TRAIN_PATH = "/bmrNAS/people/akshay/dl/abct_data/train/"
VALID_PATH = "/bmrNAS/people/akshay/dl/abct_data/train/"
TEST_PATH = "/bmrNAS/people/akshay/dl/abct_data/train/"


# Check if save path is initialized.
if not SAVE_PATH:
    raise ValueError('Set `SAVE_PATH` in defaults.py')

