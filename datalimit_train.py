import argparse
import os
import random

from keras import backend as K

import glob_constants

import config as MCONFIG
from config import DeeplabV3Config, SegnetConfig, EnsembleUDSConfig, UNetConfig, UNetMultiContrastConfig, UNet2_5DConfig, DeeplabV3_2_5DConfig
from im_generator import calc_generator_info, img_generator, img_generator_oai, get_class_freq
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS, BINARY_CROSS_ENTROPY_LOSS

from weight_classes import CLASS_FREQ_DAT_PATH

from models import get_model

from natsort import natsorted

import oai_train

import utils
import parse_pids


def data_limitation_train(config_name, vals_dict=None):
    """
    Train data limited networks
    :return:
    """

    pids = utils.load_pik(parse_pids.PID_TXT_PATH)
    pids_dict = {5: ['9003406', '9007827', '9047800', '9056363', '9068453'],
                 15: ['9003406', '9007827', '9040390', '9094865', '9172459',
                      '9047800', '9056363', '9068453', '9085290', '9087863',
                      '9102858', '9211869', '9311328', '9331465', '9279291'],
                 30: ['9003406', '9007827', '9040390', '9094865', '9172459',
                      '9192885', '9215390', '9264046', '9309170', '9382271',
                      '9047800', '9056363', '9068453', '9085290', '9087863',
                      '9102858', '9211869', '9311328', '9331465', '9332085',
                      '9352437', '9357137', '9357383', '9369649', '9444401',
                      '9493245', '9567704', '9597990', '9279291', '9596610'],
                 60: random.sample(pids, 60)}
    print('Data limitation......')
    import math
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_data_limit/oai_data'
    num_pids = len(pids)

    pid_counts = natsorted(list(pids_dict.keys()))

    for pid_count in pid_counts:
        MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_data_limit/oai_data/%03d' % pid_count

        if pid_count > num_pids:
            pid_count = num_pids

        # select pids that were sampled
        pids_sampled = pids_dict[pid_count]
        s_ratio = math.ceil(num_pids / pid_count)

        config = get_config(config_name)

        config.N_EPOCHS = math.ceil(10 * num_pids / pid_count)
        config.DROP_FACTOR = config.DROP_FACTOR ** (1 / s_ratio)
        config.PIDS = pids_sampled if pid_count != num_pids else None

        print('# Subjects: %d' % pid_count)

        if vals_dict is not None:
            for key in vals_dict.keys():
                val = vals_dict[key]
                config.set_attr(key, val)

        config.save_config()
        config.summary()

        oai_train.train_model(config)

        print('Epochs: %d' % config.N_EPOCHS)
        K.clear_session()

    # must exit because config constant has been overwritten
    exit()


SUPPORTED_MODELS = ['unet', 'segnet', 'deeplab']

if __name__=='__main__':

    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/data_limit'

    parser = argparse.ArgumentParser(description='Train OAI dataset')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
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
    print(models)
    for model in models:
        print(model)
        # Data limitation experiment: Train Unet, Deeplab, and Segnet with limited data
        if model == 'unet':
            config = UNetConfig()
            oai_train.train(config, vals_dict={'LOSS': WEIGHTED_CROSS_ENTROPY_LOSS,
                                               'INCLUDE_BACKGROUND': True})
        elif model == 'deeplab':
            config = DeeplabV3Config()
            oai_train.train(config, vals_dict={'LOSS': WEIGHTED_CROSS_ENTROPY_LOSS,
                                               'INCLUDE_BACKGROUND': True})
        elif model == 'segnet_2d':
            config = SegnetConfig()
            oai_train.train(config, vals_dict={'LOSS': WEIGHTED_CROSS_ENTROPY_LOSS,
                                                                'INCLUDE_BACKGROUND': True,
                                                                'INITIAL_LEARNING_RATE': 1e-3})
        else:
            raise ValueError('model %s not supported' % model)
