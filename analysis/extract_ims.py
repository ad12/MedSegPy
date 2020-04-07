import sys

sys.path.insert(0, '../')

from medsegpy import utils

if __name__ == '__main__':
    files = ['/Users/arjundesai/Desktop/roma_data/ismrm-data/edge/9993833_V01-Aug00_012',
             '/Users/arjundesai/Desktop/roma_data/ismrm-data/good/9993833_V01-Aug00_022',
             '/Users/arjundesai/Desktop/roma_data/ismrm-data/ml_transition/9993833_V01-Aug00_032']

    for f in files:
        utils.save_ims(f)
