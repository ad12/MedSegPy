from __future__ import print_function, division

from abc import ABC, abstractmethod
import argparse
import os

import argparse
import os
import pickle
from copy import deepcopy

import keras.callbacks as kc
import numpy as np
from keras import backend as K
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import Adam
from keras.utils import plot_model

import config as MCONFIG
import mri_utils
from cross_validation import cv_util
from generators import im_gens
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS, dice_loss, focal_loss
from models.models import get_model
from utils import io_utils, parallel_utils as putils, utils, dl_utils

import defaults

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True
FREEZE_LAYERS = None


class CommandLineInterface(ABC):
    __DESCRIPTION__ = ''  # To override in subclasses.

    # Argument keys.
    __ARG_KEY_GPU__ = 'GPU'

    def __init__(self):
        if not self.__DESCRIPTION__:
            raise ValueError('__DESCRIPTION__ must be set for all subclasses')

        self.__args = None
        self.base_parser = argparse.ArgumentParser(description=self.__DESCRIPTION__)

        # Initialize argument parser.
        self.init_parser()

    @abstractmethod
    def init_parser(self):
        pass

    def parse(self):
        print('Parsing')
        args = self.base_parser.parse_args()
        self.__args = vars(args)

    @abstractmethod
    def run(self):
        pass

    @property
    def args(self):
        return self.__args

    def get_arg(self, key):
        return self.args[key]

    def _add_gpu_argument(self, parser):
        parser.add_argument('-g', '--%s' % self.__ARG_KEY_GPU__,
                            metavar='G', type=str, nargs='?', default='-1',
                            dest=self.__ARG_KEY_GPU__,
                            help='gpu id to use. default=0')

    @property
    def gpu(self):
        self.verify_args()
        return self.args[self.__ARG_KEY_GPU__]

    def verify_args(self):
        if not self.__args:
            raise ValueError('Run parse() before calling method')


class NNTrain(CommandLineInterface):
    __DESCRIPTION__ = 'Train networks for segmentation'

    # Argument Parser
    _ARG_KEY_CONFIG = 'config'
    _ARG_KEY_K_FOLD_CROSS_VALIDATION = 'k_fold_cross_validation'
    _ARG_KEY_HO_TEST = 'ho_test'
    _ARG_KEY_HO_VALID = 'ho_valid'
    _ARG_KEY_CLASS_WEIGHTS = 'class_weights'
    _ARG_KEY_EXPERIMENT = 'experiment'
    _ARG_KEY_ABS_SAVE_PATH = 'save_path'
    _ARG_KEY_FINE_TUNE_PATH = 'fine_tune_path'
    _ARG_KEY_FREEZE_LAYERS = 'freeze_layers'
    _ARG_KEY_SAVE_ALL_WEIGHTS = 'save_all_weights'
    _ARG_KEY_SAVE_MODEL = 'save_model'

    @property
    def config(self):
        self.verify_args()
        return self.args[self._ARG_KEY_CONFIG]

    @property
    def save_best_weights(self):
        self.verify_args()
        return not self.args[self._ARG_KEY_SAVE_ALL_WEIGHTS]

    @property
    def save_model(self):
        return self.args[self._ARG_KEY_SAVE_MODEL]

    @property
    def frozen_layers(self):
        frozen_layers = self.get_arg(self._ARG_KEY_FREEZE_LAYERS)
        return utils.convert_data_type(frozen_layers, tuple) if frozen_layers else None

    @property
    def write_grads(self):
        return False

    @property
    def write_images(self):
        return False

    def init_parser(self):
        arg_subparser = self.base_parser.add_subparsers(help='supported configs for different architectures',
                                                        dest=self._ARG_KEY_CONFIG)
        subparsers = MCONFIG.init_cmd_line_parser(arg_subparser)

        for s_parser in subparsers:
            self._add_gpu_argument(s_parser)
            s_parser.add_argument('-k', '--%s' % self._ARG_KEY_K_FOLD_CROSS_VALIDATION,
                                  metavar='K', default=None, nargs='?',
                                  dest=self._ARG_KEY_K_FOLD_CROSS_VALIDATION,
                                  help='Use k-fold cross-validation for training. '
                                       'Argument is k (int) or filepath (str).')
            s_parser.add_argument('--%s' % self._ARG_KEY_HO_TEST,
                                  metavar='T', type=int, default=1, nargs='?',
                                  dest=self._ARG_KEY_HO_TEST,
                                  help='Number of hold-out test bins.')
            s_parser.add_argument('--%s' % self._ARG_KEY_HO_VALID,
                                  metavar='V', type=int, default=1, nargs='?',
                                  dest=self._ARG_KEY_HO_VALID,
                                  help='Number of hold-out validation bins.')
            s_parser.add_argument('--%s' % self._ARG_KEY_CLASS_WEIGHTS,
                                  type=tuple, nargs='?', default=None,
                                  dest=self._ARG_KEY_CLASS_WEIGHTS,
                                  help='Weight classes in order.')
            s_parser.add_argument('--%s' % self._ARG_KEY_EXPERIMENT,
                                  type=str, nargs='?', default='',
                                  dest=self._ARG_KEY_EXPERIMENT,
                                  help='Experiment to run.')
            s_parser.add_argument('--%s' % self._ARG_KEY_ABS_SAVE_PATH,
                                  type=str, nargs='?', default='',
                                  dest=self._ARG_KEY_ABS_SAVE_PATH,
                                  help='Save path. Must be absolute path.')
            s_parser.add_argument('--%s' % self._ARG_KEY_FINE_TUNE_PATH,
                                  type=str, default='', nargs='?',
                                  dest=self._ARG_KEY_FINE_TUNE_PATH,
                                  help='Directory to fine tune.')
            s_parser.add_argument('--%s' % self._ARG_KEY_FREEZE_LAYERS,
                                  type=str, default=None, nargs='?',
                                  dest=self._ARG_KEY_FREEZE_LAYERS,
                                  help='Range of layers to freeze. eg. `(0,100)`, `(5, 45)`, `(5,)`')
            s_parser.add_argument('--%s' % self._ARG_KEY_SAVE_ALL_WEIGHTS,
                                  default=False, action='store_const', const=True,
                                  dest=self._ARG_KEY_SAVE_ALL_WEIGHTS,
                                  help="Store weights at each epoch. Default: False")
            s_parser.add_argument('--%s' % self._ARG_KEY_SAVE_MODEL,
                                  default=False, action='store_const', const=True,
                                  dest=self._ARG_KEY_SAVE_MODEL,
                                  help="Save model as h5 file. Default: False")

            # add support for specifying tissues
            mri_utils.init_cmd_line(s_parser)

    def run(self):
        gpu = self.gpu
        abs_save_path = self.args[self._ARG_KEY_ABS_SAVE_PATH]
        experiment_dir = self.args[self._ARG_KEY_EXPERIMENT]
        fine_tune_dirpath = self.args[self._ARG_KEY_FINE_TUNE_PATH]
        k_fold_cross_validation = self.args[self._ARG_KEY_K_FOLD_CROSS_VALIDATION]

        # Validate either fine-tune or experiment type selected
        if not fine_tune_dirpath and not experiment_dir and not abs_save_path:
            raise ValueError('--%s,  --%s, or --%s must be specified' % (self._ARG_KEY_EXPERIMENT,
                                                                         self._ARG_KEY_FINE_TUNE_PATH,
                                                                         self._ARG_KEY_ABS_SAVE_PATH))

        if abs_save_path:
            MCONFIG.SAVE_PATH_PREFIX = abs_save_path
        else:
            MCONFIG.SAVE_PATH_PREFIX = os.path.join(defaults.SAVE_PATH, experiment_dir)
        print('OUTPUT_DIR: %s' % MCONFIG.SAVE_PATH_PREFIX)
        
        # Initialize GPUs that are visible.
        print('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Load config and corresponding config dictionary
        c = MCONFIG.get_config(config_cp_save_tag=self.args[self._ARG_KEY_CONFIG],
                               create_dirs=not fine_tune_dirpath)
        config_dict = c.parse_cmd_line(self.args)

        # Parse tissue (classes) to segment.
        config_dict['TISSUES'] = mri_utils.parse_tissues(self.args)

        if fine_tune_dirpath:
            # parse freeze layers
            self._train_fine_tune(c, config_dict)
            exit(0)

        if k_fold_cross_validation:
            self._train_cross_validation(c, config_dict)
            exit(0)

        self._train(c, config_dict)

    def _train_cross_validation(self, c, config_dict):
        k_fold_cross_validation = self.get_arg(self._ARG_KEY_K_FOLD_CROSS_VALIDATION)
        if k_fold_cross_validation.isdigit():
            k_fold_cross_validation = int(k_fold_cross_validation)

        ho_valid = self.get_arg(self._ARG_KEY_HO_VALID)
        ho_test = self.get_arg(self._ARG_KEY_HO_TEST)

        # Initialize CrossValidation wrapper
        cv_wrapper = cv_util.CrossValidationProcessor(k_fold_cross_validation,
                                                      num_valid_bins=ho_valid,
                                                      num_test_bins=ho_test)

        print('Loading %d-fold cross-validation data from %s...' % (cv_wrapper.k, cv_wrapper.filepath))

        cv_file = cv_wrapper.filepath
        cv_k = cv_wrapper.k

        cv_exp_id = 1

        base_save_path = c.CP_SAVE_PATH
        for tr_f, val_f, test_f, tr_bins, val_bins, test_bins in cv_wrapper.run():
            c.init_cross_validation(train_files=tr_f,
                                    valid_files=val_f,
                                    test_files=test_f,
                                    train_bins=tr_bins,
                                    valid_bins=val_bins,
                                    test_bins=test_bins,
                                    cv_k=cv_k,
                                    cv_file=cv_file,
                                    cp_save_path=os.path.join(base_save_path, 'cv-exp-%03d' % cv_exp_id))
            cv_exp_id += 1

            self._train(c, config_dict)

    def _train(self, config, config_param_dict=None):
        """
        Train model specified by config
        :param config: a Config object
        :param config_param_dict: a dictionary of config parameters to change (default = None)
                                  e.g. {'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}
        """

        if config_param_dict is not None:
            for key in config_param_dict.keys():
                val = config_param_dict[key]
                config.set_attr(key, val)

        config.save_config()
        config.summary()

        self._train_model(config)

        K.clear_session()

    def _train_model(self, config, optimizer=None, model=None):
        """
        Train model
        :param config: a Config object
        :param optimizer: a Keras optimizer (default = None)
        """

        # Load data from config.
        cp_save_path = config.CP_SAVE_PATH
        cp_save_tag = config.CP_SAVE_TAG
        n_epochs = config.N_EPOCHS
        pik_save_path = config.PIK_SAVE_PATH
        loss = config.LOSS
        class_weights = self.get_arg(self._ARG_KEY_CLASS_WEIGHTS)

        if model is None:
            model = get_model(config)

        # Plot model to png file.
        #plot_model(model,
        #           to_file=os.path.join(cp_save_path, 'model.png'),
        #           show_shapes=True)

        # If initial weight path specified, initialize model with weights.
        if config.INIT_WEIGHT_PATH:
            print('Initializing with weights: %s' % config.INIT_WEIGHT_PATH)
            model.load_weights(config.INIT_WEIGHT_PATH)
            frozen_layers = self.frozen_layers
            if frozen_layers:
                if len(frozen_layers) == 1:
                    fl = range(FREEZE_LAYERS[0], len(model.layers))
                else:
                    fl = range(frozen_layers[0], frozen_layers[1])
                print('freezing layers %s' % fl)
                for i in fl:
                    model.layers[i].trainable = False

        # Replicate model on multiple gpus - note this does not solve issue of having too large of a model
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if num_gpus > 1:
            print('Running multi gpu model')
            model = putils.ModelMGPU(model, gpus=num_gpus)

        # If no optimizer is provided, default to Adam
        # TODO (arjundd): Add support for addtional optimizers
        if optimizer is None:
            optimizer = Adam(lr=config.INITIAL_LEARNING_RATE, beta_1=0.99, beta_2=0.995, epsilon=1e-8,
                             decay=config.ADAM_DECAY, amsgrad=config.USE_AMSGRAD)

        # Track learning rate on tensorboard.
        loss_func = get_training_loss(loss, weights=class_weights)
        lr_metric = self._learning_rate_callback(optimizer)
        model.compile(optimizer=optimizer, loss=loss_func, metrics=[lr_metric, dice_loss])

        # Set image format to be (N, dim1, dim2, dim3, channel).
        K.set_image_data_format('channels_last')

        # Define model callbacks.
        cp_cb = ModelCheckpoint(os.path.join(cp_save_path, cp_save_tag + '_weights.{epoch:03d}-{val_loss:.4f}.h5'),
                                save_best_only=self.save_best_weights)
        tfb_cb = tfb(config.TF_LOG_DIR,
                     write_grads=self.write_grads,
                     write_images=self.write_images)
        hist_cb = LossHistory()

        callbacks_list = [tfb_cb, cp_cb, hist_cb]

        # Step decay for learning rate
        if config.USE_STEP_DECAY:
            lr_cb = lrs(self._step_decay_callback(config.INITIAL_LEARNING_RATE,
                                                  config.MIN_LEARNING_RATE,
                                                  config.DROP_FACTOR,
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

        train_gen = generator.img_generator(state=im_gens.GeneratorState.TRAINING)
        val_gen = generator.img_generator(state=im_gens.GeneratorState.VALIDATION)

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

        model_json = model.to_json()
        model_json_save_path = os.path.join(config.CP_SAVE_PATH, 'model.json')
        with open(model_json_save_path, "w") as json_file:
            json_file.write(model_json)

        if self.save_model:
            model.save(filepath=os.path.join(config.CP_SAVE_PATH, 'model.h5'), overwrite=True)

    def _train_fine_tune(self, config, vals_dict=None):
        dirpath = self.get_arg(self._ARG_KEY_FINE_TUNE_PATH)

        # Initialize for fine tuning.
        config.load_config(os.path.join(dirpath, 'config.ini'))

        # Get best weight path.
        best_weight_path = dl_utils.get_weights(dirpath)
        print('Best weight path: %s' % best_weight_path)

        config.init_fine_tune(best_weight_path)

        # Only load command line arguments that are not the default.
        temp_config = type(config)(create_dirs=False)
        if vals_dict is not None:
            for key in vals_dict.keys():
                val = vals_dict[key]
                val_default = getattr(temp_config, key)
                if val != val_default:
                    config.set_attr(key, val)

        config.save_config()
        config.summary()

        self._train_model(config)

        K.clear_session()

    def _learning_rate_callback(self, optimizer):
        """
        Wrapper for learning rate tensorflow metric
        :param optimizer: a Keras optimizer
        :return: a Tensorflow callback
        """

        def lr(y_true, y_pred):
            return optimizer.lr

        return lr

    def _step_decay_callback(self, initial_lr, min_lr, drop_factor, drop_rate):
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
            if lrate < min_lr:
                lrate = min_lr

            return lrate

        return step_decay

    def parse(self):
        super().parse()
        if len(self.gpu.split(',')) > 1 and self.save_model:
            raise ValueError('Model cannot be saved when using multiple gpus for training.')


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


if __name__ == '__main__':
    nn_train = NNTrain()
    nn_train.parse()
    nn_train.run()
