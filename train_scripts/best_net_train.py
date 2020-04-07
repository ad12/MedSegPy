import argparse
import logging
import os

import numpy as np

from medsegpy import glob_constants, oai_train, config as MCONFIG
from medsegpy.config import DeeplabV3Config
from medsegpy.modeling.losses import WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))

CLASS_WEIGHTS = np.asarray([1, 1 / 5])


if __name__ == '__main__':
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/best_network'

    parser = argparse.ArgumentParser(description='Train OAI dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    args = parser.parse_args()
    logger.info(args)
    gpu = args.gpu
    glob_constants.SEED = args.seed

    logger.info(glob_constants.SEED)

    logger.info('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # oai_train.fine_tune('/bmrNAS/people/arjun/msk_seg_networks/best_network/deeplabv3_2d/2018-11-27-00-40-24/', DeeplabV3Config(), vals_dict={'INITIAL_LEARNING_RATE':8e-6})
    # oai_train.train(DeeplabV3Config(), vals_dict={'AUGMENT_DATA': False, 'N_EPOCHS': 100, 'LOSS': BINARY_CROSS_ENTROPY_SIG_LOSS})
    # oai_train.train(DeeplabV3Config(), vals_dict={'AUGMENT_DATA': True, 'N_EPOCHS': 50, 'LOSS': BINARY_CROSS_ENTROPY_SIG_LOSS})
    oai_train.train(DeeplabV3Config(),
                    {'N_EPOCHS': 100, 'TRAIN_BATCH_SIZE': 12, 'USE_STEP_DECAY': False,
                     'AUGMENT_DATA': False, 'LOSS': WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS},
                    class_weights=CLASS_WEIGHTS)
    # oai_train.fine_tune(dirpath='/bmrNAS/people/arjun/msk_seg_networks/best_network/deeplabv3_2d/pretrained/',
    #                     config=DeeplabV3Config(create_dirs=False),
    #                     vals_dict={'N_EPOCHS': 100, 'TRAIN_BATCH_SIZE': 12, 'USE_STEP_DECAY': False,
    #                                'AUGMENT_DATA': False, 'LOSS': WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS,
    #                                'INITIAL_LEARNING_RATE': 8e-6},
    #                     class_weights=CLASS_WEIGHTS
    #                     )
