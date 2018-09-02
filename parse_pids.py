import os
import utils


PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug'
PID_TXT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/train_pid.dat'

# Create list of pids
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
