import argparse
import os

import config as MCONFIG
import glob_constants
import oai_train
from config import DeeplabV3Config, SegnetConfig, UNetConfig
from losses import WEIGHTED_CROSS_ENTROPY_LOSS, BINARY_CROSS_ENTROPY_LOSS, BINARY_CROSS_ENTROPY_SIG_LOSS, FOCAL_LOSS
import losses

SUPPORTED_MODELS = ['unet', 'segnet', 'deeplab']

if __name__ == '__main__':

    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/loss_limit'

    parser = argparse.ArgumentParser(description='Train OAI dataset')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0', help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    parser.add_argument('-m', '--model', metavar='M', nargs=1, choices=SUPPORTED_MODELS)
    parser.add_argument('-bce', action='store_const', default=False, const=True)
    parser.add_argument('-bcse', action='store_const', default=False, const=True)
    parser.add_argument('-focal', nargs='?', default=None, const=3.0, type=float)
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
    include_background = True
    loss_func = BINARY_CROSS_ENTROPY_LOSS if args.bce else WEIGHTED_CROSS_ENTROPY_LOSS
    if args.bcse:
        loss_func = BINARY_CROSS_ENTROPY_SIG_LOSS
        include_background = False

    if args.focal:
        loss_func = FOCAL_LOSS
        include_background = False
        losses.FOCAL_LOSS_GAMMA = args.focal
        print(losses.FOCAL_LOSS_GAMMA)

    for model in models:
        # Data limitation experiment: Train Unet, Deeplab, and Segnet with limited data
        if model == 'unet':
            config = UNetConfig()
            oai_train.train(config, vals_dict={'LOSS': loss_func,
                                               'INCLUDE_BACKGROUND': include_background})
        elif model == 'deeplab':
            config = DeeplabV3Config()
            oai_train.train(config, vals_dict={'LOSS': loss_func,
                                               'INCLUDE_BACKGROUND': include_background})
        elif model == 'segnet':
            config = SegnetConfig()
            oai_train.train(config, vals_dict={'LOSS': loss_func,
                                               'INCLUDE_BACKGROUND': include_background})
        else:
            raise ValueError('model %s not supported' % model)
