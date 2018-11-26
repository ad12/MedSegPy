import argparse
import os

import config as MCONFIG
import glob_constants
import oai_train
from config import UNet2_5DConfig

SUPPORTED_MODELS = ['unet']

if __name__ == '__main__':

    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/volume_limited'

    parser = argparse.ArgumentParser(description='Train OAI dataset')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0', help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    parser.add_argument('-m', '--model', metavar='M', nargs=1, choices=SUPPORTED_MODELS)
    parser.add_argument('-ds', metavar='S', type=int, default=3, nargs='?', help='stack depth of input (i.e. # slices)')
    parser.add_argument('-ft', nargs='?', metavar='PATH', type=str, default=None, help='fine tune model from path')

    args = parser.parse_args()
    print(args)
    gpu = args.gpu

    fine_tune_path = args.ft

    models = args.model

    num_slices = args.ds

    glob_constants.SEED = args.seed

    print(glob_constants.SEED)

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    for model in models:
        # Volume limitation experiment: does 2.5D network make a difference? 3 slices? 5 slices?
        # The data was augmented slice by slice, so we can't just put slices together
        # therefore, we train on non-augmented data as the difference in performance between non-augmented and augmented
        # was not significant
        if model == 'unet':
            config = UNet2_5DConfig()
            config.IMG_SIZE = (config.IMG_SIZE[0], config.IMG_SIZE[1], num_slices)
            if fine_tune_path is not None:
                oai_train.fine_tune(fine_tune_path, config, {'INITIAL_LEARNING_RATE': 1e-4})
            else:
                oai_train.train(config, {})
        else:
            raise ValueError('model %s not supported' % model)
