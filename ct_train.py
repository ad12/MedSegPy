import os
import pickle
import numpy as np
import h5py

import logging
from keras import backend as K
from keras.callbacks import LearningRateScheduler as lrs
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard as tfb
from keras.optimizers import Adam

from generators import im_gens
from losses import get_training_loss, WEIGHTED_CROSS_ENTROPY_LOSS, dice_loss, focal_loss
from models.models import get_model
from utils import io_utils, parallel_utils as putils, utils, dl_utils
from utils.logger import setup_logger

import defaults

import config as MCONFIG
from nn_train import LossHistory, NNTrain

from config import Config

CLASS_WEIGHTS = np.asarray([100, 1])
SAVE_BEST_WEIGHTS = True
FREEZE_LAYERS = None


class CTGenerator(im_gens.OAIGenerator):
    """
        Generator to be used with files where training/testing data is written as 2D slices
        Filename: PATIENTID_VISIT_AUGMENTATION-NUMBER_SLICE-NUMBER
        Filename Format: '%07d_V%02d_Aug%02d_%03d' (e.g. '0000001_V00_Aug00_001.h5
    """
    SUPPORTED_TAGS = ['abct', 'ct']
    __EXPECTED_IMG_SIZE_DIMS__ = 2

    def __init__(self, config: Config, windows=None):
        if windows and config.num_neighboring_slices() != len(windows):
            raise ValueError("Expected {} windows".format(config.num_neighboring_slices()))
        self.windows = windows
        super().__init__(config)

    def __load_inputs__(self, data_path: str, file: str):
        im, seg = self._load_inputs_basic(data_path, file)
        im = self._preprocess(im, self.windows[0] if self.windows else None)
        return im, seg

    def _load_inputs_basic(self, data_path: str, file: str):
        im_path = '%s/%s.im' % (data_path, file)
        with h5py.File(im_path, 'r') as f:
            im = f['data'][:]
            if len(im.shape) == 2:
                im = im[..., np.newaxis]

        seg_path = '%s/%s.seg' % (data_path, file)
        with h5py.File(seg_path, 'r') as f:
            seg = f['data'][:].astype('float32')

        assert len(im.shape) == 3
        assert len(seg.shape) == 4 and seg.shape[-2] == 1

        return im, seg

    def _preprocess(self, im, window):
        # Apply windowing.
        if window:
            im = np.clip(im, window[0], window[1])

        # Preprocess by max normalizing.
        im -= np.min(im)
        im /= np.max(im)

        return im

    def __load_neighboring_slices__(self, num_slices, filepath, max_slice):
        """Stacks 2D CT slices from single patient clipped at different window levels.

        Overloads traditional 2.5D networks that look at neighboring slices.
        """
        data_path, filename = os.path.dirname(filepath), os.path.basename(filepath)
        im, seg = self._load_inputs_basic(data_path, filename)
        h, w = im.shape[:2]

        ims = []
        for window in self.windows:
            ims.append(np.squeeze(self._preprocess(np.copy(im), window)))
        im = np.stack(ims, axis=-1)

        assert im.shape == (h, w, self.config.num_neighboring_slices())
        return im, seg


class CTTrain(NNTrain):
    __DESCRIPTION__ = 'Train networks for ct segmentation'

    _ARG_KEY_WINDOWS = "windows"

    def parse_windows(self, windows):
        windowing = {"soft": (400, 50),
                     "bone": (1800, 400),
                     "liver": (150, 30),
                     "spine": (250, 50),
                     "custom": (500, 50)}
        vals = []
        for w in windows:
            if w not in windowing:
                raise KeyError("Window {} not found".format(w))
            window_width = windowing[w][0]
            window_level = windowing[w][1]
            upper = window_level + window_width / 2
            lower = window_level - window_width / 2

            vals.append((lower, upper))

        return vals

    def init_parser(self):
        arg_subparser = self.base_parser.add_subparsers(help='supported configs for different architectures',
                                                        dest=self._ARG_KEY_CONFIG)
        subparsers = MCONFIG.init_cmd_line_parser(arg_subparser)

        for s_parser in subparsers:
            self.__add_gpu__argument__(s_parser)
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
            s_parser.add_argument('--%s' % self._ARG_KEY_WINDOWS,
                                  metavar='W', type=str, nargs='*',
                                  dest=self._ARG_KEY_WINDOWS,
                                  help='(min, max) windows for clipping data')

            # add support for specifying tissues
            s_parser.add_argument("--classes", type=int, nargs="+", required=True,
                                  help="tissue indices to segment")

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

        # Load config and corresponding config dictionary
        c = MCONFIG.get_config(config_cp_save_tag=self.args[self._ARG_KEY_CONFIG],
                               create_dirs=not fine_tune_dirpath)
        config_dict = c.parse_cmd_line(self.args)
        # Parse tissue (classes) to segment.
        config_dict['TISSUES'] = self.args["classes"]

        # Initialize logger.
        setup_logger(c.CP_SAVE_PATH, name=__name__)
        logger = logging.getLogger(__name__)
        logger.info("Args:\n{}".format(self.args))
        logger.info('OUTPUT_DIR: %s' % c.CP_SAVE_PATH)

        # Initialize GPUs that are visible.
        logger.info('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        if fine_tune_dirpath:
            # parse freeze layers
            self.__train_fine_tune(c, config_dict)
            exit(0)

        if k_fold_cross_validation:
            self.__train_cross_validation(c, config_dict)
            exit(0)

        self.__train(c, config_dict)

    def __train_cross_validation(self, c, config_dict):
        raise NotImplementedError("Cross validation not supported for CT training")

    def __train_model(self, config, optimizer=None, model=None):
        """
        Train model
        :param config: a Config object
        :param optimizer: a Keras optimizer (default = None)
        """
        logger = logging.getLogger(__name__)
        # Load data from config.
        cp_save_path = config.CP_SAVE_PATH
        cp_save_tag = config.CP_SAVE_TAG
        n_epochs = config.N_EPOCHS
        pik_save_path = config.PIK_SAVE_PATH
        loss = config.LOSS
        class_weights = self.get_arg(self._ARG_KEY_CLASS_WEIGHTS)

        if model is None:
            model = get_model(config)

        # If initial weight path specified, initialize model with weights.
        if config.INIT_WEIGHT_PATH:
            logger.info('Initializing with weights: %s' % config.INIT_WEIGHT_PATH)
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

        # Replicate model on multiple gpus - note this does not solve issue of having too large of a model
        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        if num_gpus > 1:
            logger.info('Running multi gpu model')
            model = putils.ModelMGPU(model, gpus=num_gpus)

        # If no optimizer is provided, default to Adam
        # TODO (arjundd): Add support for addtional optimizers
        if optimizer is None:
            optimizer = Adam(lr=config.INITIAL_LEARNING_RATE, beta_1=0.99, beta_2=0.995, epsilon=1e-8,
                             decay=config.ADAM_DECAY, amsgrad=config.USE_AMSGRAD)

        # Track learning rate on tensorboard.
        loss_func = get_training_loss(loss, weights=class_weights)
        lr_metric = self.__learning_rate_callback(optimizer)
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
            lr_cb = lrs(self.__step_decay_callback(config.INITIAL_LEARNING_RATE,
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

        windows = self.parse_windows(self.get_arg(self._ARG_KEY_WINDOWS))
        generator = self.build_generator(config, windows)
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

    @classmethod
    def build_generator(cls, cfg, windows):
        return CTGenerator(cfg, windows)


if __name__ == '__main__':
    nn_train = CTTrain()
    nn_train.parse()
    nn_train.run()
