import argparse
import os

import config as MCONFIG
import glob_constants
import oai_train
from config import UNetConfig

SUPPORTED_MODELS = ['unet']

if __name__ == '__main__':

    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/augment_limited'

    parser = argparse.ArgumentParser(description='Train OAI dataset')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0', help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    parser.add_argument('-m', '--model', metavar='M', nargs=1, choices=SUPPORTED_MODELS)
    parser.add_argument('-a', action='store_const', default=False, const=True)
    parser.add_argument('-ft', nargs='?', metavar='PATH', type=str, default=None, help='fine tune model from path')

    args = parser.parse_args()
    print(args)
    gpu = args.gpu

    fine_tune_path = args.ft

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
            if fine_tune_path is not None:
                oai_train.fine_tune(fine_tune_path, config, {'AUGMENT_DATA': False, 'N_EPOCHS': 100,
                                                             'USE_STEP_DECAY': False, 'INITIAL_LEARNING_RATE': 1e-5,
                                                             'DROP_FACTOR': (0.8) ** (1 / 5)})
            else:
                oai_train.train(config, vals_dict={'AUGMENT_DATA': False, 'N_EPOCHS': 100, 'DROP_FACTOR': (0.8) ** (1 / 5)})

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
