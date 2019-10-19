SAVE_PATH = '/bmrNAS/people/arjun/msk_seg_networks/msk_feat_comp/'  # Put absolute path for where data should be stored
TRAIN_PATH = '/bmrNAS/people/arjun/data/oai_data/h5_files_2d/train'
VALID_PATH = '/bmrNAS/people/arjun/data/oai_data/h5_files_2d/valid/'
TEST_PATH = '/bmrNAS/people/arjun/data/oai_data/h5_files_2d/test'


# Check if save path is initialized.
if not SAVE_PATH:
    raise ValueError('Set `SAVE_PATH` in defaults.py')

