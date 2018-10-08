import os

import utils

TRAIN_PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug'
VAL_PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid'
TEST_PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'
PID_TXT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/train_pid.dat'


def get_pids(pid_path):
    files = os.listdir(pid_path)
    pids = []
    for m_file in files:
        # extract pid
        filename_split = m_file.split('_')
        pid = filename_split[0]
        pids.append(pid)

    pids = list(set(pids))
    return pids


# Create list of pids
if __name__ == '__main__':
    train_pids = get_pids(TRAIN_PID_PATH)
    utils.save_pik(train_pids, PID_TXT_PATH)

    test_pids = get_pids(TEST_PID_PATH)
    val_pids = get_pids(VAL_PID_PATH)

    # print length of training testing and validation
    print('Train: %d subjects' % len(train_pids))
    print('Val: %d subjects' % len(val_pids))
    print('Test: %d subjects' % len(test_pids))
