import logging
import os

from medsegpy.utils import io_utils
from medsegpy.utils.logger import setup_logger

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))

TRAIN_PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug'
VAL_PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid'
TEST_PID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'
PID_TXT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/train_pid.dat'
PID_LOG_PATH = '/bmrNAS/people/arjun/msk_seg_networks/train_pid_log.txt'


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
    # Initialize logger.
    setup_logger(PID_LOG_PATH)
    logger.info("Saving to: {}".format(PID_TXT_PATH))

    train_pids = get_pids(TRAIN_PID_PATH)
    io_utils.save_pik(train_pids, PID_TXT_PATH)

    test_pids = get_pids(TEST_PID_PATH)
    val_pids = get_pids(VAL_PID_PATH)

    # Log length of training testing and validation
    logger.info('Train: %d subjects' % len(train_pids))
    logger.info('Val: %d subjects' % len(val_pids))
    logger.info('Test: %d subjects' % len(test_pids))
