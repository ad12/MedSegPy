import argparse
import os

import config as MCONFIG
import glob_constants
import oai_train
from config import DeeplabV3Config
from losses import BINARY_CROSS_ENTROPY_SIG_LOSS, WEIGHTED_CROSS_ENTROPY_LOSS, FOCAL_LOSS
import numpy as np

CLASS_WEIGHTS = np.asarray([100, 1])


if __name__ == '__main__':
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/best_network'

    parser = argparse.ArgumentParser(description='Train OAI dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    args = parser.parse_args()
    print(args)
    gpu = args.gpu
    glob_constants.SEED = args.seed

    print(glob_constants.SEED)

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    #oai_train.fine_tune('/bmrNAS/people/arjun/msk_seg_networks/best_network/deeplabv3_2d/2018-11-27-00-40-24/', DeeplabV3Config(), vals_dict={'INITIAL_LEARNING_RATE':8e-6})
    #oai_train.train(DeeplabV3Config(), vals_dict={'AUGMENT_DATA': False, 'N_EPOCHS': 100, 'LOSS': BINARY_CROSS_ENTROPY_SIG_LOSS})
    # oai_train.train(DeeplabV3Config(), vals_dict={'AUGMENT_DATA': True, 'N_EPOCHS': 50, 'LOSS': BINARY_CROSS_ENTROPY_SIG_LOSS})
    # oai_train.train(DeeplabV3Config(), {'N_EPOCHS': 100, 'TRAIN_BATCH_SIZE': 12, 'USE_STEP_DECAY': False,
    #                                     'AUGMENT_DATA': False, 'LOSS': WEIGHTED_CROSS_ENTROPY_LOSS,
    #                                     'INCLUDE_BACKGROUND': True}, class_weights=CLASS_WEIGHTS)
    oai_train.fine_tune(dirpath='/bmrNAS/people/arjun/msk_seg_networks/best_network/deeplabv3_2d/2018-11-27-00-40-24/',
                        config=DeeplabV3Config(create_dirs=False),
                        vals_dict={'N_EPOCHS': 100, 'TRAIN_BATCH_SIZE': 12, 'USE_STEP_DECAY': False,
                                   'AUGMENT_DATA': False, 'LOSS': FOCAL_LOSS,
                                   'INITIAL_LEARNING_RATE': 8e-6}
                        )
