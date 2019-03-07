# Author: Arjun Desai, arjun.desai@duke.edu, 2018 June

from __future__ import print_function, division

import argparse
import os
import pickle

import keras.callbacks as kc
import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import Adam
from keras.utils import plot_model

import config as MCONFIG
import glob_constants
from cross_validation import cv_utils
from generators import im_gens
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS, dice_loss, focal_loss
from models.models import get_model
from utils import io_utils, parallel_utils as putils, utils
import mri_utils

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True

def train_model(config, optimizer=None, model=None, class_weights=None):
    """
    Train model
    :param config: a Config object
    :param optimizer: a Keras optimizer (default = None)
    """

    # Load data from config
    cp_save_path = config.CP_SAVE_PATH
    cp_save_tag = config.CP_SAVE_TAG
    n_epochs = config.N_EPOCHS
    pik_save_path = config.PIK_SAVE_PATH
    loss = config.LOSS

    if model is None:
        model = get_model(config)

    # Fine tune - initialize with weights
    if config.FINE_TUNE:
        print('loading weights')
        model.load_weights(config.INIT_WEIGHT_PATH, by_name=True)

    # Replicate model on multiple gpus - note this does not solve issue of having too large of a model
    num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    if num_gpus > 1:
        print('Running multi gpu model')
        model = putils.ModelMGPU(model, gpus=num_gpus)

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
                  loss=loss_func, metrics=[lr_metric, dice_loss, focal_loss()])

    # set image format to be (N, dim1, dim2, dim3, ch)
    K.set_image_data_format('channels_last')

    # model callbacks
    cp_cb = ModelCheckpoint(os.path.join(cp_save_path, cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5'),
                            save_best_only=SAVE_BEST_WEIGHTS)
    tfb_cb = tfb(config.TF_LOG_DIR,
                 write_grads=False,
                 write_images=False)
    hist_cb = LossHistory()

    callbacks_list = [tfb_cb, cp_cb, hist_cb]

    # Step decay for learning rate
    if config.USE_STEP_DECAY:
        lr_cb = lrs(step_decay_wrapper(config.INITIAL_LEARNING_RATE, config.MIN_LEARNING_RATE, config.DROP_FACTOR,
                                       config.DROP_RATE))
        callbacks_list.append(lr_cb)

    # use early stopping
    if config.USE_EARLY_STOPPING:
        es_cb = EarlyStopping(monitor=config.EARLY_STOPPING_CRITERION,
                              min_delta=config.EARLY_STOPPING_MIN_DELTA,
                              patience=config.EARLY_STOPPING_PATIENCE)
        callbacks_list.append(es_cb)

    generator = im_gens.get_generator(config)
    generator.summary()

    train_nbatches, valid_nbatches = generator.num_steps()

    train_gen = generator.img_generator(state='training')
    val_gen = generator.img_generator(state='validation')

    # plot model
    plot_model(model, to_file=os.path.join(cp_save_path, 'model.png'), show_shapes=True)

    # Start training
    model.fit_generator(train_gen,
                        train_nbatches,
                        epochs=n_epochs,
                        validation_data=val_gen,
                        validation_steps=valid_nbatches,
                        callbacks=callbacks_list,
                        verbose=1)

    # Save optimizer state
    io_utils.save_optimizer(model.optimizer, config.CP_SAVE_PATH)

    # Save files to write as output
    data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
    with open(pik_save_path, "wb") as f:
        pickle.dump(data, f)

    # # Save model
    # model.save(filepath=os.path.join(config.CP_SAVE_PATH, 'model.h5'), overwrite=True)


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


EXP_DIR_MAP = {'arch': 'architecture_limited',
               'aug': 'augment_limited',
               'best': 'best_network',
               'data': 'data_limit',
               'loss': 'loss_limit',
               'vol': 'volume_limited',
               'control': 'control_exps'
               }

if __name__ == '__main__':
    base_parser = argparse.ArgumentParser(description='Train OAI dataset')
    arg_subparser = base_parser.add_subparsers(help='supported configs for different architectures', dest='config')
    subparsers = MCONFIG.init_cmd_line_parser(arg_subparser)

    for s_parser in subparsers:
        s_parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                            help='gpu id to use. default=0')
        s_parser.add_argument('-s', '--seed', metavar='S', type=int, nargs='?', default=None,
                            help='python seed to initialize filter weights. default=None')
        s_parser.add_argument('-k', '--k_fold_cross_validation', metavar='K', type=int, default=None, nargs='?',
                            help='Use k-fold cross-validation for training. Argument specifies k')
        s_parser.add_argument('--ho_test', metavar='T', type=int, default=1, nargs='?',
                            help='Number of hold-out test bins')
        s_parser.add_argument('--ho_valid', metavar='V', type=int, default=1, nargs='?',
                            help='Number of hold-out validation bins')
        s_parser.add_argument('--class_weights', type=tuple, nargs='?', default=CLASS_WEIGHTS,
                            help='weight classes in order')
        s_parser.add_argument('--experiment', type=str, nargs=1,
                              choices=sorted(list(EXP_DIR_MAP.keys())),
                              help='experiment to run'
                              )
        s_parser.add_argument('--fine_tune_path', type=str, default='', nargs='?',
                              help='directory to fine tune.')

        s_parser.add_argument('--save_all_weights', default=False, action='store_const', const=True,
                              help="store weights for each epoch. Default: False")

        # add support for specifying tissues
        mri_utils.init_cmd_line(s_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    experiment_filepath = EXP_DIR_MAP[args.experiment[0]]
    MCONFIG.SAVE_PATH_PREFIX = os.path.join('/bmrNAS/people/arjun/msk_seg_networks', experiment_filepath)

    #MCONFIG.SAVE_PATH_PREFIX = os.path.join('./sample_data/cmd_line', experiment_filepath)

    gpu = args.gpu
    glob_constants.SEED = args.seed
    k_fold_cross_validation = args.k_fold_cross_validation

    fine_tune_dirpath = args.fine_tune_path
    print(glob_constants.SEED)

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    SAVE_BEST_WEIGHTS = not args.save_all_weights

    c = MCONFIG.get_config(config_name = vargin['config'])
    config_dict = c.parse_cmd_line(vargin)

    # parse tissues
    config_dict['TISSUES'] = mri_utils.parse_tissues(vargin)

    if fine_tune_dirpath:
        fine_tune(fine_tune_dirpath, c, config_dict)
        exit(0)

    if k_fold_cross_validation:
        ho_test = args.ho_test
        ho_valid = args.ho_valid

        bins_files = cv_utils.load_cross_validation(k_fold_cross_validation)
        bins_split = cv_utils.get_cv_experiments(k_fold_cross_validation, num_valid_bins=ho_valid, num_test_bins=ho_test)
        cv_exp_id = 1
        for bin_inds in bins_split:
            train_files, valid_files, test_files = cv_utils.get_fnames(bins_files, bin_inds)
            c.init_cross_validation(train_files, valid_files, test_files, 'cv-exp-%03d' % cv_exp_id)
            cv_exp_id += 1

            train(c, config_dict)
    else:
        train(c, config_dict)
