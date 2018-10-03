import argparse
import os

import glob_constants

import config as MCONFIG
from config import DeeplabV3Config, SegnetConfig, UNetConfig
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS, BINARY_CROSS_ENTROPY_LOSS

import oai_train

SUPPORTED_MODELS = ['unet']


if __name__=='__main__':

    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/augment_limited'

    parser = argparse.ArgumentParser(description='Train OAI dataset')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0', help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    parser.add_argument('-m', '--model', metavar='M', nargs=1, choices=SUPPORTED_MODELS)
    parser.add_argument('-a', action='store_const', default=False, const=True)

    args = parser.parse_args()
    print(args)
    gpu = args.gpu

    models = args.model
    if args.a:
        models = SUPPORTED_MODELS

    glob_constants.SEED = args.seed

    print(glob_constants.SEED)

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    for model in models:
        # Data limitation experiment: Train Unet, Deeplab, and Segnet with limited data
        if model == 'unet':
            config = UNetConfig()
            oai_train.train(config, vals_dict={'AUGMENT_DATA': False, 'N_EPOCHS': 100})

        # elif model == 'deeplab':
        #     config = DeeplabV3Config()
        #     oai_train.train(config, vals_dict={'LOSS': WEIGHTED_CROSS_ENTROPY_LOSS,
        #                                        'INCLUDE_BACKGROUND': True})
        # elif model == 'segnet_2d':
        #     config = SegnetConfig()
        #     oai_train.train(config, vals_dict={'LOSS': WEIGHTED_CROSS_ENTROPY_LOSS,
        #                                        'INCLUDE_BACKGROUND': True})
        else:
            raise ValueError('model %s not supported' % model)