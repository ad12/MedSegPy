from abc import ABC, abstractmethod
import argparse
from copy import deepcopy
import logging
import os
import pickle
from typing import Tuple, Union

os.environ["MSK_SEG_NETWORKS_PROJECT"] = "tech-considerations_v3"

import keras.callbacks as kc
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard as tfb
import numpy as np

from medsegpy import glob_constants, config as MCONFIG, solver, \
    nn_test
from medsegpy.cross_validation import cv_util
from medsegpy.data import im_gens, data_loader
from medsegpy.losses import get_training_loss, dice_loss
from medsegpy.modeling import get_model
from medsegpy.utils import dl_utils, mri_utils
from medsegpy.utils import utils, io_utils, parallel_utils as putils
from medsegpy.utils.logger import setup_logger
from medsegpy.oai_test import test_dir

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True
FREEZE_LAYERS = None


class CommandLineInterface(ABC):
    __DESCRIPTION__ = ''  # To override in subclasses.

    # Argument keys.
    __ARG_KEY_GPU__ = 'gpu_id'
    __ARG_KEY_NUM_GPU__ = 'num_gpus'

    def __init__(self):
        if not self.__DESCRIPTION__:
            raise ValueError('__DESCRIPTION__ must be set for all subclasses')

        self.__args = None
        self.base_parser = argparse.ArgumentParser(
            description=self.__DESCRIPTION__)

        # Initialize argument parser.
        self.init_parser()

    @abstractmethod
    def init_parser(self):
        pass

    def parse(self):
        logger.info('Parsing')
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
        parser.add_argument('--{}'.format(self.__ARG_KEY_NUM_GPU__),
                            default=1,
                            type=int,
                            dest=self.__ARG_KEY_NUM_GPU__,
                            help="number of gpus to use. defaults to 1")
        parser.add_argument('--%s' % self.__ARG_KEY_GPU__,
                            metavar='G', type=int, nargs='*', default=[],
                            dest=self.__ARG_KEY_GPU__,
                            help='gpu id to use. defaults to []')

    @property
    def gpu(self):
        """GPU ids in str format - eg. `'0,1'` (gpus 0 and 1), '-1' (cpu), etc."""
        self.verify_args()

        args_gpu_ids = self.args[self.__ARG_KEY_GPU__]
        args_num_gpus = self.args[self.__ARG_KEY_NUM_GPU__]
        gpu_ids = args_gpu_ids if args_gpu_ids else dl_utils.get_available_gpus(
            args_num_gpus)
        gpu_ids_tf_str = ",".join([str(g_id) for g_id in gpu_ids])

        return gpu_ids_tf_str

    @property
    def num_gpus(self):
        return self.args[self.__ARG_KEY_NUM_GPU__]

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
    _ARG_KEY_EXPERIMENT = 'experiment'
    _ARG_KEY_OUTPUT_DIR = 'output_dir'
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
        return utils.convert_data_type(frozen_layers,
                                       tuple) if frozen_layers else None

    @property
    def write_grads(self):
        return False

    @property
    def write_images(self):
        return False

    @staticmethod
    def _add_classes_parser(parser):
        mri_utils.init_cmd_line(parser)

    def _parse_classes(self):
        return mri_utils.parse_tissues(self.args)

    def _add_default_args(self, parser):
        parser.add_argument('-k',
                            '--%s' % self._ARG_KEY_K_FOLD_CROSS_VALIDATION,
                            metavar='K', default=None, nargs='?',
                            dest=self._ARG_KEY_K_FOLD_CROSS_VALIDATION,
                            help='Use k-fold cross-validation. '
                                 'Argument is k (int) or filepath (str).')
        parser.add_argument('--%s' % self._ARG_KEY_HO_TEST,
                            metavar='T', type=int, default=1, nargs='?',
                            dest=self._ARG_KEY_HO_TEST,
                            help='Number of hold-out test bins.')
        parser.add_argument('--%s' % self._ARG_KEY_HO_VALID,
                            metavar='V', type=int, default=1, nargs='?',
                            dest=self._ARG_KEY_HO_VALID,
                            help='Number of hold-out validation bins.')
        parser.add_argument('--%s' % self._ARG_KEY_EXPERIMENT,
                            type=str, nargs='?', default='',
                            dest=self._ARG_KEY_EXPERIMENT,
                            help='Experiment to run.')
        parser.add_argument('--%s' % self._ARG_KEY_OUTPUT_DIR,
                            type=str, nargs='?', default='',
                            dest=self._ARG_KEY_OUTPUT_DIR,
                            help='Output dir')
        parser.add_argument('--%s' % self._ARG_KEY_FINE_TUNE_PATH,
                            type=str, default='', nargs='?',
                            dest=self._ARG_KEY_FINE_TUNE_PATH,
                            help='Directory to fine tune.')
        parser.add_argument('--%s' % self._ARG_KEY_FREEZE_LAYERS,
                            type=str, default=None, nargs='?',
                            dest=self._ARG_KEY_FREEZE_LAYERS,
                            help='Range of layers to freeze. eg. `(0,100)`, `(5, 45)`, `(5,)`')
        parser.add_argument('--%s' % self._ARG_KEY_SAVE_ALL_WEIGHTS,
                            default=False, action='store_const',
                            const=True,
                            dest=self._ARG_KEY_SAVE_ALL_WEIGHTS,
                            help="Store weights at each epoch. Default: False")
        parser.add_argument('--%s' % self._ARG_KEY_SAVE_MODEL,
                            default=False, action='store_const',
                            const=True,
                            dest=self._ARG_KEY_SAVE_MODEL,
                            help="Save model as h5 file. Default: False")

    def init_parser(self):
        self.base_parser.add_argument(
            "--config-file",
            default=None,
            help="config file to load fields from"
        )
        arg_subparser = self.base_parser.add_subparsers(
            help='supported configs for different architectures',
            dest=self._ARG_KEY_CONFIG)
        subparsers = MCONFIG.init_cmd_line_parser(arg_subparser)

        for s_parser in subparsers:
            self._add_gpu_argument(s_parser)
            self._add_default_args(s_parser)
            self._add_classes_parser(s_parser)

    def init_config(self):
        output_dir = self.args[self._ARG_KEY_OUTPUT_DIR]
        config_file = self.args["config_file"]

        config = MCONFIG.get_config(
            config_cp_save_tag=self.args[self._ARG_KEY_CONFIG],
            create_dirs=False,
            output_dir=output_dir,
        )

        # Load from args.
        print("Loading config from {}".format(config_file))
        config_dict = config.parse_cmd_line(self.args)
        config_dict['TISSUES'] = self._parse_classes()
        if config_dict is not None:
            for key in config_dict.keys():
                val = config_dict[key]
                config.set_attr(key, val)

        # Load from file.
        if config_file:
            print("Loading config from {}".format(config_file))
            config.load_config(config_file)

        if not config.OUTPUT_DIR:
            raise ValueError("config.OUTPUT_DIR not defined")

        return config

    def run(self):
        gpu = self.gpu
        k_fold_cross_validation = self.args[
            self._ARG_KEY_K_FOLD_CROSS_VALIDATION]

        # Initialize GPUs that are visible.
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        # Load config and corresponding config dictionary
        config = self.init_config()

        # Initialize logger.
        output_dir = config.OUTPUT_DIR
        setup_logger(output_dir)
        logger.info("Args:\n{}".format(self.args))
        logger.info("Using {} GPU(s): {}".format(self.num_gpus, gpu))
        logger.info("OUTPUT_DIR: {}".format(output_dir))

        # TODO: Add fine_tuning support
        # if fine_tune_dirpath:
        #     # parse freeze layers
        #     self._train_fine_tune(c, config_dict)
        #     exit(0)

        if k_fold_cross_validation:
            self._train_cross_validation(config)
            exit(0)

        self._train(config)

    def _train_cross_validation(self, config):
        k_fold_cross_validation = self.get_arg(
            self._ARG_KEY_K_FOLD_CROSS_VALIDATION)
        if k_fold_cross_validation.isdigit():
            k_fold_cross_validation = int(k_fold_cross_validation)

        ho_valid = self.get_arg(self._ARG_KEY_HO_VALID)
        ho_test = self.get_arg(self._ARG_KEY_HO_TEST)

        # Initialize CrossValidation wrapper
        cv_wrapper = cv_util.CrossValidationProcessor(k_fold_cross_validation,
                                                      num_valid_bins=ho_valid,
                                                      num_test_bins=ho_test)

        logger.info('Loading %d-fold cross-validation data from %s...' % (
            cv_wrapper.k, cv_wrapper.filepath))

        cv_file = cv_wrapper.filepath
        cv_k = cv_wrapper.k

        cv_exp_id = 1

        output_dir = config.OUTPUT_DIR
        for tr_f, val_f, test_f, tr_bins, val_bins, test_bins in cv_wrapper.run():
            config_copy = config.init_cross_validation(
                train_files=tr_f,
                valid_files=val_f,
                test_files=test_f,
                train_bins=tr_bins,
                valid_bins=val_bins,
                test_bins=test_bins,
                cv_k=cv_k,
                cv_file=cv_file,
                output_dir=os.path.join(
                    output_dir,
                    'cv-exp-%03d' % cv_exp_id)
            )
            cv_exp_id += 1

            self._train(config_copy)

    def _train_fine_tune(self, config, vals_dict = None):
        dirpath = self.get_arg(self._ARG_KEY_FINE_TUNE_PATH)

        # Initialize for fine tuning.
        config.load_config(os.path.join(dirpath, 'config.ini'))

        # Get best weight path.
        best_weight_path = dl_utils.get_weights(dirpath)
        logger.info('Best weight path: %s' % best_weight_path)

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

    def _train(self, config):
        """Train model specified by config.

        Args:
            config (Config): A config.
        """
        config.save_config()
        config.summary()

        self._train_model(config)

        K.clear_session()

        self._test(config)

    def _test(self, config):
        logger.info("Beginning testing...")
        config = deepcopy(config)  # will be modified below.
        dirpath = config.OUTPUT_DIR

        # By default, h5 data is saved and voxel spacing is automatically
        # determined.
        test_params = {
            "batch_size": config.TEST_BATCH_SIZE,
        }
        test_dir(
            dirpath,
            vals_dict=nn_test.create_config_dict(test_params),
            save_h5_data=True,
            voxel_spacing=None,
        )

        K.clear_session()

    def _init_model(self, config, model):
        """Initialize model with weights and apply any freezing necessary."""
        logger.info(
            'Loading weights from {}'.format(config.INIT_WEIGHT_PATH)
        )
        model.load_weights(config.INIT_WEIGHT_PATH)
        frozen_layers = self.frozen_layers
        if frozen_layers:
            if len(frozen_layers) == 1:
                fl = range(FREEZE_LAYERS[0], len(model.layers))
            else:
                fl = range(frozen_layers[0], frozen_layers[1])
            logger.info('freezing layers %s' % fl)
            for i in fl:
                model.layers[i].trainable = False

    def _build_data_loaders(
        self,
        config
    ) -> Union[Tuple[im_gens.Generator, im_gens.Generator],
               data_loader.DataLoader]:
        """Builds train and val data loaders.
        """
        generator = im_gens.get_generator(config)
        # try:
        #     train_gen = data_loader.get_data_loader(
        #         config,
        #         state=im_gens.GeneratorState.TRAINING,
        #         shuffle=True,
        #         drop_last=True,
        #         generator=generator,
        #     )
        #     val_gen = data_loader.get_data_loader(
        #         config,
        #         state=im_gens.GeneratorState.VALIDATION,
        #         shuffle=False,
        #         drop_last=False,
        #         generator=generator,
        #     )
        #     train_nbatches, valid_nbatches = None, None
        #     use_multiprocessing = False
        #     return train_gen, val_gen
        # except ValueError as e:
        # logger.info("{}\nDefaulting to traditional generator".format(e))
        return generator, generator

    def _train_model(self, config, optimizer = None, model = None):
        """Train model

        Args:
            config (Config): The config for training.
            optimizer: A Keras-compatible optimizer.
            model: If `config.INIT_WEIGHT_PATH` is specified, weights will be
                loaded into the model.
        """

        # Load data from config.
        n_epochs = config.N_EPOCHS
        loss = config.LOSS
        class_weights = config.CLASS_WEIGHTS
        num_workers = config.NUM_WORKERS

        # Set global constants.
        glob_constants.SEED = config.SEED

        if model is None:
            model = get_model(config)
        logger.info(model.summary())

        if config.INIT_WEIGHT_PATH:
            self._init_model(config, model)

        # Replicate model on multiple gpus.
        # Note this does not solve issue of having too large of a model
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if num_gpus > 1:
            logger.info('Running multi gpu model')
            model = putils.ModelMGPU(model, gpus=num_gpus)

        if optimizer is None:
            optimizer = solver.build_optimizer(config)

        # TODO: Add more options for metrics.
        loss_func = get_training_loss(loss, weights=class_weights)
        lr_metric = self._learning_rate_callback(optimizer)
        model.compile(
            optimizer=optimizer,
            loss=loss_func,
            metrics=[lr_metric, dice_loss]
        )

        # Set image format to be (N, dim1, dim2, dim3, channel).
        K.set_image_data_format('channels_last')

        # Define model callbacks.
        cp_cb = ModelCheckpoint(
            os.path.join(
                config.OUTPUT_DIR,
                "weights.{epoch:03d}-{val_loss:.4f}.h5"
            ),
            save_best_only=self.save_best_weights,
            save_weights_only=True,
        )
        tfb_cb = tfb(
            config.OUTPUT_DIR,
            write_grads=self.write_grads,
            write_images=self.write_images,
        )
        hist_cb = LossHistory()
        callbacks_list = [tfb_cb, cp_cb, hist_cb]

        if config.LR_SCHEDULER_NAME:
            callbacks_list.append(solver.build_lr_scheduler(config))

        if config.USE_EARLY_STOPPING:
            es_cb = EarlyStopping(
                monitor=config.EARLY_STOPPING_CRITERION,
                min_delta=config.EARLY_STOPPING_MIN_DELTA,
                patience=config.EARLY_STOPPING_PATIENCE
            )
            callbacks_list.append(es_cb)

        train_loader, val_loader = self._build_data_loaders(config)
        if isinstance(train_loader, im_gens.Generator):
            train_nbatches, valid_nbatches = train_loader.num_steps()
            train_loader.summary()
            train_loader = train_loader.img_generator(
                state=im_gens.GeneratorState.TRAINING
            )
            val_loader = val_loader.img_generator(
                state=im_gens.GeneratorState.VALIDATION
            )
            use_multiprocessing = False
        elif isinstance(train_loader, data_loader.DataLoader):
            train_nbatches, valid_nbatches = None, None
            logger.info("Training Summary:\n" + train_loader.summary())
            logger.info("Validation Summary:\n" + val_loader.summary())
            use_multiprocessing = True
        else:
            raise ValueError(
                "Unknown data loader {}".format(type(train_loader))
            )

        # Start training
        model.fit_generator(
            train_loader,
            train_nbatches,
            epochs=n_epochs,
            validation_data=val_loader,
            validation_steps=valid_nbatches,
            callbacks=callbacks_list,
            workers=num_workers,
            use_multiprocessing=use_multiprocessing,
            verbose=1
        )

        # Save optimizer state
        io_utils.save_optimizer(model.optimizer, config.OUTPUT_DIR)

        # Save files to write as output
        data = [hist_cb.epoch, hist_cb.losses, hist_cb.val_losses]
        pik_data_path = os.path.join(config.OUTPUT_DIR, "pik_data.dat")
        with open(pik_data_path, "wb") as f:
            pickle.dump(data, f)

        model_json = model.to_json()
        model_json_save_path = os.path.join(config.OUTPUT_DIR, 'model.json')
        with open(model_json_save_path, "w") as json_file:
            json_file.write(model_json)

        if self.save_model:
            model.save(filepath=os.path.join(config.OUTPUT_DIR, 'model.h5'),
                       overwrite=True)

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

    def parse(self):
        super().parse()
        if len(self.gpu.split(',')) > 1 and self.save_model:
            raise ValueError(
                'Model cannot be saved when using multiple gpus for training.'
            )


class LossHistory(kc.Callback):
    """
    A Keras callback to log training history
    """

    def on_train_begin(self, logs = {}):
        self.val_losses = []
        self.losses = []
        # self.lr = []
        self.epoch = []

    def on_epoch_end(self, batch, logs = {}):
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
        # self.lr.append(step_decay(len(self.losses)))
        self.epoch.append(len(self.losses))


if __name__ == '__main__':
    nn_train = NNTrain()
    nn_train.parse()
    nn_train.run()
