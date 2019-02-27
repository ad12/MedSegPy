# Author: Arjun Desai, arjun.desai@duke.edu, 2018 June

from __future__ import print_function, division

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
from config import DeeplabV3Config, UNetConfig, SegnetConfig, init_cmd_line_parser, parse_cmd_line, SUPPORTED_CONFIGS
from im_generator import calc_generator_info, img_generator, img_generator_oai
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS, dice_loss
from models import get_model
from cross_validation import cv_utils

CLASS_WEIGHTS = np.asarray([100, 1])


def train_model(config, optimizer=None, model=None, class_weights=None):
    """
    Train model
    :param config: a Config object
    :param optimizer: a Keras optimizer (default = None)
    """

    # Load data from config
    train_path = config.TRAIN_PATH
    valid_path = config.VALID_PATH
    train_batch_size = config.TRAIN_BATCH_SIZE
    valid_batch_size = config.VALID_BATCH_SIZE
    cp_save_path = config.CP_SAVE_PATH
    cp_save_tag = config.CP_SAVE_TAG
    n_epochs = config.N_EPOCHS
    file_types = config.FILE_TYPES
    pik_save_path = config.PIK_SAVE_PATH
    tag = config.TAG
    learn_files = config.LEARN_FILES
    loss = config.LOSS
    layers_to_freeze = []

    # Get model based on config
    img_size = config.IMG_SIZE

    if model is None:
        model = get_model(config)

    # Fine tune - initialize with weights
    if (config.FINE_TUNE):
        print('loading weights')
        model.load_weights(config.INIT_WEIGHT_PATH, by_name=True)

    # If no optimizer is provided, default to Adam
    if optimizer is None:
        optimizer = Adam(lr=config.INITIAL_LEARNING_RATE, beta_1=0.99, beta_2=0.995, epsilon=1e-8,
                         decay=config.ADAM_DECAY, amsgrad=config.USE_AMSGRAD)

    # Load loss function
    # if weighted cross entropy, load weights
    if loss == WEIGHTED_CROSS_ENTROPY_LOSS and class_weights is None:
        # print('calculating freq')
        # freq_file = CLASS_FREQ_DAT_WEIGHTS_AUG if config.AUGMENT_DATA else CLASS_FREQ_DAT_WEIGHTS_NO_AUG
        # print('Weighting with file: %s' % freq_file)
        # class_freqs = utils.load_pik(freq_file)
        # class_weights = get_class_weights(class_freqs)
        # class_weights = np.reshape(class_weights, (1, 2))
        print(class_weights)

    loss_func = get_training_loss(loss, weights=class_weights)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer,
                  loss=loss_func, metrics=[lr_metric, dice_loss])

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')
    train_files, train_nbatches = calc_generator_info(train_path, train_batch_size, learn_files=learn_files,
                                                      pids=config.PIDS, augment_data=config.AUGMENT_DATA)
    valid_files, valid_nbatches = calc_generator_info(valid_path, valid_batch_size)

    print('INFO: Train size: %d, batch size: %d' % (len(train_files), train_batch_size))
    print('INFO: Valid size: %d, batch size: %d' % (len(valid_files), valid_batch_size))
    print('INFO: Image size: %s' % (img_size,))
    print('INFO: Image types included in training: %s' % (file_types,))
    print('INFO: Number of frozen layers: %s' % len(layers_to_freeze))

    # model callbacks
    cp_cb = ModelCheckpoint(
        os.path.join(cp_save_path, cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5'),
        save_best_only=True)
    tfb_cb = tfb(config.TF_LOG_DIR,
                 write_grads=False,
                 write_images=False)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb]

    # Step decay for learning rate
    if (config.USE_STEP_DECAY):
        lr_cb = lrs(step_decay_wrapper(config.INITIAL_LEARNING_RATE, config.MIN_LEARNING_RATE, config.DROP_FACTOR,
                                       config.DROP_RATE))
        callbacks_list.append(lr_cb)

    # Determine training generator based on version of config
    if (config.VERSION > 1):
        train_gen = img_generator_oai(train_path,
                                      train_batch_size,
                                      config=config,
                                      state='training',
                                      shuffle_epoch=True)
        val_gen = img_generator_oai(valid_path,
                                    valid_batch_size,
                                    config=config,
                                    state='validation',
                                    shuffle_epoch=False)
    else:
        train_gen = img_generator(train_path, train_batch_size, img_size, tag, config.TISSUES, pids=config.PIDS)
        val_gen = img_generator(valid_path, valid_batch_size, img_size, tag, config.TISSUES)

    # Start training
    model.fit_generator(
        train_gen,
        train_nbatches,
        epochs=n_epochs,
        validation_data=val_gen,
        validation_steps=valid_nbatches,
        callbacks=callbacks_list,
        verbose=1)

    # Save optimizer state
    utils.save_optimizer(model.optimizer, config.CP_SAVE_PATH)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    # Save model
    model.save(filepath=os.path.join(config.CP_SAVE_PATH, 'model.h5'), overwrite=True)


def get_class_weights(freqs):
    # weight by median and scale to 1
    weights = np.median(freqs) / freqs
    weights = weights / np.min(weights)

    return weights


def get_lr_metric(optimizer):
    """
    Wrapper for learning rate tensorflow metric
    :param optimizer: a Keras optimizer
    :return: a Tensorflow callback
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


def step_decay_wrapper(initial_lr=1e-4, min_lr=1e-8, drop_factor=0.8, drop_rate=1.0):
    """
    Wrapper for learning rate step decay
    :param initial_lr: initial learning rate (default = 1e-4)
    :param min_lr: minimum learning rate (default = None)
    :param drop_factor: factor to drop (default = 0.8)
    :param drop_rate: rate of learning rate drop (default = 1.0 epochs)
    :return: a Tensorflow callback
    """
    initial_lr = initial_lr
    drop_factor = drop_factor
    drop_rate = drop_rate
    min_lr = min_lr

    def step_decay(epoch):
        import math
        lrate = initial_lr * math.pow(drop_factor, math.floor((1 + epoch) / drop_rate))

        if (lrate < min_lr):
            lrate = min_lr

        return lrate

    return step_decay


class LossHistory(kc.Callback):
    """
    A Keras callback to log training history
    """

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.losses = []
        # self.lr = []
        self.epoch = []

    def on_epoch_end(self, batch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
        # self.lr.append(step_decay(len(self.losses)))
        self.epoch.append(len(self.losses))


def fine_tune(dirpath, config, vals_dict=None, class_weights=None):
    # # If a fine-tune directory already exits, skip this directory
    # if (os.path.isdir(os.path.join(dirpath, 'fine_tune'))):
    #     print('Skipping %s - fine_tune folder exists' % dirpath)

    # Initialize for fine tuning
    config.load_config(os.path.join(dirpath, 'config.ini'))

    # Get best weight path
    best_weight_path = utils.get_weights(dirpath)
    print('Best weight path: %s' % best_weight_path)

    config.init_fine_tune(best_weight_path)
    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.summary()
    config.save_config()

    train_model(config, class_weights=class_weights)

    K.clear_session()


def train_debug():
    print('')
    print('DEBUGGING.....')
    config = DeeplabV3Config()
    config.DEBUG = True
    config.N_EPOCHS = 1
    config.OS = 16
    config.DIL_RATES = (1, 1, 1)

    config.save_config()

    train_model(config)

    K.clear_session()


def unet_2d_multi_contrast_train():
    """
    Train multi contrast network
    """
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/akshay/dl/oai_data/segnet_2d/'

    # By default, loads weights from original 2D unet
    config = UNetMultiContrastConfig()

    # By default, loads weights from original 2D unet
    # To not load these weights by default, uncomment line below
    # config.INIT_UNET_2D = False

    # Adjust hyperparameters
    config.N_EPOCHS = 25
    config.DROP_FACTOR = 0.8
    config.DROP_RATE = 1.0

    train_model(config)

    # need to exit because overwritting config parameters
    exit()


def train(config, vals_dict=None, class_weights=CLASS_WEIGHTS):
    """
    Train config after applying vals_dict
    :param config: a Config object
    :param vals_dict: a dictionary of config parameters to change (default = None)
                      e.g. {'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}
    """

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.save_config()
    config.summary()

    train_model(config, class_weights=class_weights)

    K.clear_session()


def get_config(name):
    configs = [DeeplabV3Config(create_dirs=False), UNetConfig(create_dirs=False), SegnetConfig(create_dirs=False)]

    for config in configs:
        if config.CP_SAVE_TAG == name:
            c = config
            c.init_training_paths(c.DATE_TIME_STR)
            return c

    raise ValueError('config %s not found' % name)


if __name__ == '__main__':
    MCONFIG.SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/architecture_limit'

    parser = argparse.ArgumentParser(description='Train OAI dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use. default=0')
    parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None,
                        help='python seed to initialize filter weights. default=None')
    parser.add_argument('-k', '--k_cross_validation', metavar='K', type=int, default=None, nargs='?',
                        help='Use k-fold cross-validation for training. Argument specifies k')
    parser.add_argument('-ho_test', metavar='T', type=int, default=1, nargs='?',
                        help='Number of hold-out test bins')
    parser.add_argument('-ho_valid', metavar='V', type=int, default=1, nargs='?',
                        help='Number of hold-out validation bins')
    parser.add_argument('--model', type=str, nargs=1, choices=SUPPORTED_CONFIGS,
                        help='model to use')

    init_cmd_line_parser(parser)

    args = parser.parse_args()
    vargin = vars(args)

    gpu = args.gpu
    glob_constants.SEED = args.seed
    k_cross_validation = args.k_cross_validation

    print(glob_constants.SEED)

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    config_dict = parse_cmd_line(vargin)
    c = get_config(args.model)

    if k_cross_validation:
        ho_test = args.ho_test
        ho_valid = args.ho_valid

        bins_files = cv_utils.load_cross_validation(k_cross_validation)
        bins_split = cv_utils.get_cv_experiments(k_cross_validation, num_valid_bins=ho_valid, num_test_bins=ho_test)
        cv_exp_id = 1
        for bin_inds in bins_split:
            train_files, valid_files, test_files = cv_utils.get_fnames(bins_files, bin_inds)
            c.init_cross_validation(train_files, valid_files, test_files, 'cv-exp-%03d' % cv_exp_id)
            cv_exp_id += 1

            train(c, config_dict)
    else:
        train(c, config_dict)