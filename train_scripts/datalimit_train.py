import argparse
import os
import random

from keras import backend as K
from natsort import natsorted

import config as MCONFIG
import glob_constants
import oai_train
import parse_pids
import utils
from config import DeeplabV3Config, SegnetConfig, UNetConfig


def get_config(name):
    if name == 'unet':
        return UNetConfig()
    elif name == 'segnet':
        return SegnetConfig()
    elif name == 'deeplab':
        return DeeplabV3Config()
    else:
        raise ValueError('config %s not supported' % name)


def data_limitation_train(config_name, vals_dict=None, pc=None):
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
    num_pids = len(pids)

    pid_counts = natsorted(list(pids_dict.keys()))

    if pc is not None:
        pid_counts = [pc]

    for pid_count in pid_counts:

        MCONFIG.SAVE_PATH_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_seg_networks/data_limit', '%03d' % pid_count)

        if pid_count > num_pids:
            pid_count = num_pids

        # select pids that were sampled
        pids_sampled = pids_dict[pid_count]
        s_ratio = math.ceil(num_pids / pid_count)

        config = get_config(config_name)

        config.N_EPOCHS = math.ceil(100 * num_pids / pid_count)
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


SUPPORTED_MODELS = ['unet', 'segnet', 'deeplab']
SUPPORTED_PATIENT_COUNTS = [5, 15, 30, 60]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OAI dataset')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None)
    parser.add_argument('-m', '--model', metavar='M', nargs=1, choices=SUPPORTED_MODELS)
    parser.add_argument('-a', action='store_const', default=False, const=True)

    parser.add_argument('-r', '--repeat', metavar='R', nargs='?', type=int, default=1,
                        help='number of times to repeat specified experiments')
    parser.add_argument('-pc', nargs='?', type=int, default=None, choices=SUPPORTED_PATIENT_COUNTS,
                        help='specific number of patients to do experiment')

    args = parser.parse_args()
    print(args)
    gpu = args.gpu

    models = args.model
    if args.a:
        models = SUPPORTED_MODELS

    glob_constants.SEED = args.seed

    repeat_count = args.repeat
    if repeat_count < 1:
        raise ValueError('\'-r\', \'--repeat\' must be at least 1')

    patient_count = args.pc

    print(glob_constants.SEED)

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print(models)

    vals_dict = {'AUGMENT_DATA': False, 'DROP_FACTOR': 0.8 ** (1 / 5)}
    vals_dict = None

    for c in range(repeat_count):
        for model in models:
            print(model)
            # Data limitation experiment: Train Unet, Deeplab, and Segnet with limited data
            if model == 'unet':
                data_limitation_train(model, vals_dict=vals_dict, pc=patient_count)
            elif model == 'deeplab':
                data_limitation_train(model, vals_dict=vals_dict, pc=patient_count)
            elif model == 'segnet':
                data_limitation_train(model, vals_dict=vals_dict, pc=patient_count)
            else:
                raise ValueError('model %s not supported' % model)

    # data_limitation_train('unet_2d',
    #                       vals_dict={'INITIAL_LEARNING_RATE': 0.02, 'DROP_RATE': 1, 'TRAIN_BATCH_SIZE': 12})  # unet
    # data_limitation_train('deeplabv3_2d', vals_dict={'OS': 16, 'DIL_RATES': (2, 4, 6)})  # deeplab
    # data_limitation_train('segnet_2d', vals_dict={'INITIAL_LEARNING_RATE': 1e-3})  # segnet
