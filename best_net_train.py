import argparse
import os
import pickle

import keras.callbacks as kc
import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import Adam

import config as MCONFIG
import glob_constants
import utils
from config import UNetConfig, DeeplabV3Config, UNetMultiContrastConfig, SegnetConfig, DeeplabV3_2_5DConfig
from im_generator import calc_generator_info, img_generator, img_generator_oai
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS
from models import get_model
from weight_classes import CLASS_FREQ_DAT_PATH

import oai_train


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

    oai_train.train(DeeplabV3_2_5DConfig(), vals_dict={'IMG_SIZE': (288, 288, 5)})