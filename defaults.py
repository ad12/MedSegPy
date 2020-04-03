import os

# Change to update based on the project you are working on.
_PROJECT = os.environ["MSK_SEG_NETWORKS_PROJECT"]

if _PROJECT == "tech-considerations_v1" or _PROJECT == "tech-considerations_v2":
    SAVE_PATH = "/bmrNAS/people/arjun/msk_seg_networks/"
    TRAIN_PATH = "/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug"
    VALID_PATH = "/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid"
    TEST_PATH = "/bmrNAS/people/akshay/dl/oai_data/unet_2d/test"
elif _PROJECT == "tech-considerations_v3":
    SAVE_PATH = "/bmrNAS/people/arjun/results/tech-considerations"
    TRAIN_PATH = "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/train"
    VALID_PATH = "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/valid"
    TEST_PATH = "/bmrNAS/people/arjun/data/oai_data/h5_files_2d/test"
elif _PROJECT == "abCT":
    SAVE_PATH = "/bmrNAS/people/arjun/results/abCT"
    TRAIN_PATH = "/bmrNAS/people/akshay/dl/abct_data/train"
    VALID_PATH = "/bmrNAS/people/akshay/dl/abct_data/train"
    TEST_PATH = "/bmrNAS/people/akshay/dl/abct_data/train"
else:
    raise ValueError("Project {} not supported".format(_PROJECT))

# Check if save path is initialized.
if not SAVE_PATH:
    raise ValueError('Set `SAVE_PATH` in defaults.py')
