PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/train_aug_2d'
PID_TXT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/train_pid.dat'

import utils
import os
if __name__ == '__main__':
    files = os.listdir(PID_PATH)
    pids = []
    for m_file in files:
        # extract pid
        filename_split = m_file.split('_')
        pid = filename_split[0]
        pids.append(pid)
    
    pids = list(set(pids))
    
    utils.save_pik(pids, PID_TXT_PATH)
