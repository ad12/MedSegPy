import ast
import configparser
import copy
import os
from itertools import groupby
import logging
from typing import Any, Tuple
import yaml

from fvcore.common.file_io import PathManager

from medsegpy.cross_validation import cv_util
from medsegpy.data import MetadataCatalog
from medsegpy.losses import DICE_LOSS, CMD_LINE_SUPPORTED_LOSSES, get_training_loss_from_str
from medsegpy.utils import utils as utils, io_utils

logger = logging.getLogger(__name__)

# Keys that have been deprecated.
DEPRECATED_KEYS = ['NUM_CLASSES', 'TRAIN_FILES_CV', 'VALID_FILES_CV',
                   'TEST_FILES_CV', "USE_STEP_DECAY",
                   "PIK_SAVE_PATH_DIR", "PIK_SAVE_PATH", "TF_LOG_DIR",
                   "TRAIN_PATH", "VALID_PATH", "TEST_PATH",
                   "PLOT_MODEL_PATH", "FINE_TUNE", "LEARN_FILES",
                   "DEBUG",
                   "TEST_RESULT_PATH", "TEST_RESULTS_FOLDER_NAME",
                   ]
RENAMED_KEYS = {
    "CP_SAVE_PATH": "OUTPUT_DIR",
    "CP_SAVE_TAG": "MODEL_NAME",
    "INIT_WEIGHTS": "INIT_WEIGHTS",
    "TISSUES": "CATEGORIES",
}


class Config(object):
    """A config object that is 1-to-1 with supported models.

    Each subclass of :class:`Config` corresponds to a specific model
    architecture.
    """
    VERSION = 7

    # Model name specific to config. Cannot be changed.
    MODEL_NAME = ""

    # Loss function in form (id, output_mode)
    LOSS = DICE_LOSS
    CLASS_WEIGHTS = None

    # PIDS to include, None = all pids
    PIDS = None

    # Training and validation image size
    IMG_SIZE = (288, 288, 1)

    # Training parameters
    N_EPOCHS = 100
    AUGMENT_DATA = False

    # Step Decay params
    INITIAL_LEARNING_RATE = 1e-4
    LR_SCHEDULER_NAME = ""
    MIN_LEARNING_RATE = 1e-8
    DROP_FACTOR = 0.7
    DROP_RATE = 1.0
    LR_MIN_DELTA = 1e-4
    LR_PATIENCE = 0
    LR_COOLDOWN = 0
    NUM_GRAD_STEPS = 1

    # ADAM optimizer decay
    ADAM_DECAY = 0.0
    USE_AMSGRAD = False

    # Early stopping criterion
    USE_EARLY_STOPPING = False
    EARLY_STOPPING_MIN_DELTA = 0.0
    EARLY_STOPPING_PATIENCE = 0
    EARLY_STOPPING_CRITERION = 'val_loss'

    # Batch sizes
    TRAIN_BATCH_SIZE = 12
    VALID_BATCH_SIZE = 35
    TEST_BATCH_SIZE = 72

    # Categories
    CATEGORIES = []
    INCLUDE_BACKGROUND = False

    # File Types
    FILE_TYPES = ['im']

    # Transfer Learning
    INIT_WEIGHTS = ''
    FREEZE_LAYERS = ()

    # Dataset names
    TRAIN_DATASET = ""
    VAL_DATASET = ""
    TEST_DATASET = ""

    # Cross-Validation-Parameters
    USE_CROSS_VALIDATION = False
    CV_FILE = ''
    CV_K = 0
    CV_TRAIN_BINS = []
    CV_VALID_BINS = []
    CV_TEST_BINS = []
    __CV_TRAIN_FILES__ = None
    __CV_VALID_FILES__ = None
    __CV_TEST_FILES__ = None

    # Training Model Paths
    OUTPUT_DIR = ""

    # Dataset tag - What dataset are we training on? 'dess' or 'oai'
    # choose from oai_aug, oai_aug_3d
    TAG = 'oai_aug'

    # Weights kernel initializer.
    KERNEL_INITIALIZER = 'he_normal'

    # System params
    NUM_WORKERS = 1
    SEED = None

    # Evaluation params
    TEST_WEIGHT_PATH = ''
    TEST_METRICS = ["DSC", "VOE", "ASSD", "CV"]

    # Extra parameters related to different parameters.
    PREPROCESSING_WINDOWS = ()

    def __init__(self, cp_save_tag, state='training', create_dirs=True):
        if state not in ['testing', 'training']:
            raise ValueError('state must either be \'training\' or \'testing\'')

        self.MODEL_NAME = cp_save_tag
        self.STATE = state

    def init_cross_validation(self, train_files, valid_files, test_files,
                              train_bins, valid_bins, test_bins,
                              cv_k, cv_file, output_dir):
        """Initialize config for cross validation.

        Returns:
            Config: A deep copy of the config. This copy is initialized for
                cross validation.
        """
        assert self.STATE == 'training', (
            "Initializing cross-validation must be done in training state"
        )

        config = copy.deepcopy(self)
        config.USE_CROSS_VALIDATION = True
        config.CV_TRAIN_BINS = train_bins
        config.CV_VALID_BINS = valid_bins
        config.CV_TEST_BINS = test_bins
        config.CV_K = cv_k
        config.CV_FILE = cv_file

        config.__CV_TRAIN_FILES__ = train_files
        config.__CV_VALID_FILES__ = valid_files
        config.__CV_TEST_FILES__ = test_files

        config.OUTPUT_DIR = io_utils.check_dir(output_dir)

        return config

    def save_config(self):
        """Save params of config to ini file.
        """
        members = [
            attr for attr in dir(self)
            if not callable(getattr(self, attr))
               and not attr.startswith("__")
               and not (hasattr(type(self), attr) and isinstance(getattr(type(self), attr), property))
        ]
        
        filepath = os.path.join(self.OUTPUT_DIR, "config.ini")
        config_vars = dict()
        for m_var in members:
            config_vars[m_var] = getattr(self, m_var)

        # Save config
        config = configparser.ConfigParser(config_vars)
        with PathManager.open(filepath, 'w+') as configfile:
            config.write(configfile)

        logger.info("Full config saved to {}".format(os.path.abspath(filepath)))

    def _parse_special_attributes(
        self,
        full_key: str,
        value: Any
    ) -> Tuple[str, Any]:
        """Special parsing values for attributes.

        Used when loading config from a file or from list.

        Args:
            full_key (str): Upper case attribute representation.
            value (Any): Corresponding value.
        """
        if full_key in ("TRAIN_PATH", "VALID_PATH", "TEST_PATH"):
            # Ignore empty values.
            mapping = {
                "TRAIN_PATH": "TRAIN_DATASET",
                "VALID_PATH": "VAL_DATASET",
                "TEST_PATH": "TEST_DATASET",
            }
            if value:
                prev_key, prev_val = full_key, value
                value = MetadataCatalog.convert_path_to_dataset(value)
                full_key = mapping[full_key]
                logger.info("Converting {} -> {}: {} -> {}".format(
                    prev_key, full_key, prev_val, value
                ))
        elif full_key == "LOSS" and isinstance(value, str):
            value = get_training_loss_from_str(value)
        elif full_key == "OUTPUT_DIR":
            value = PathManager.get_local_path(value)

        return full_key, value

    def merge_from_file(self, cfg_filename):
        """Load a ini or yaml config file and merge it with this object.

        "MODEL_NAME" must be specified in the file.

        Args:
            cfg_filename: File path to yaml or ini file.
        """
        vars_dict = self._load_dict_from_file(cfg_filename)

        # TODO: Handle cp save tag as a protected key.
        if vars_dict['MODEL_NAME'] != self.MODEL_NAME:
            raise ValueError(
                'Wrong config. Expected {}'.format(vars_dict['MODEL_NAME'])
            )

        for full_key, value in vars_dict.items():
            full_key = str(full_key).upper()
            full_key, value = self._parse_special_attributes(
                full_key,
                value
            )

            if full_key in DEPRECATED_KEYS:
                logger.warning(
                    "Key {} is deprecated, not loading".format(full_key)
                )
                continue
            if full_key in RENAMED_KEYS:
                new_name = RENAMED_KEYS[full_key]
                logger.warning(
                    "Key {} has been renamed to {}".format(
                        full_key, new_name
                    )
                )
                full_key = new_name

            if not hasattr(self, full_key):
                raise ValueError("Key {} does not exist.".format(full_key))

            value = _check_and_coerce_cfg_value_type(
                value,
                self.__getattribute__(full_key),
                full_key
            )
            self._decode_cfg_value(value, type(self.__getattribute__(full_key)))

            # Loading config
            self.__setattr__(full_key, value)

    def merge_from_list(self, cfg_list):
        """Merge config (keys, values) in a list (e.g. from command line).

        For example, cfg_list = ['FOO_BAR', 0.5, 'BAR_FOO', (0,3,4)]
        """
        _error_with_logging(
            len(cfg_list) % 2 == 0,
            "Override list has odd length: {}; it must be a list of pairs".format(
                cfg_list
            ),
        )

        for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if full_key == "MODEL_NAME":
                raise ValueError("Cannot change key MODEL_NAME")
            if self.key_is_deprecated(full_key):
                continue

            if self.key_is_renamed(full_key):
                self.raise_key_rename_error(full_key)

            _error_with_logging(
                hasattr(self, full_key),
                "Non-existent key: {}".format(full_key),
                error_type=KeyError,
            )
            value = self._decode_cfg_value(
                v,
                type(self.__getattribute__(full_key))
            )
            value = _check_and_coerce_cfg_value_type(
                value,
                self.__getattribute__(full_key),
                full_key
            )
            self.__setattr__(full_key, value)

    @classmethod
    def _decode_cfg_value(cls, value, data_type):
        """
        Decodes a raw config value (e.g., from a yaml config files or command
        line argument) into a Python object.

        If the value is a dict, it will be interpreted as a new CfgNode.
        If the value is a str, it will be evaluated as literals.
        Otherwise it is returned as-is.
        """
        # Configs parsed from raw yaml will contain dictionary keys that need to be
        # converted to CfgNode objects
        """
        Convert string to relevant data type
        :param var_string: variable as a string (e.g.: '[0]', '1', '2.0', 'hellow')
        :param data_type: the type of the data
        :return: string converted to data_type
        """
        if not isinstance(value, str):
            return value

        if data_type is str:
            return str(value)
        elif data_type is float:
            return float(value)
        elif data_type is int:
            return int(value)
        else:
            return ast.literal_eval(value)

    def key_is_deprecated(self, full_key):
        """Test if a key is deprecated."""
        if full_key in DEPRECATED_KEYS:
            logger.warning(
                "Deprecated config key (ignoring): {}".format(full_key)
            )
            return True
        return False

    def key_is_renamed(self, full_key):
        """Test if a key is renamed."""
        return full_key in RENAMED_KEYS

    def raise_key_rename_error(self, full_key):
        new_key = RENAMED_KEYS[full_key]
        if isinstance(new_key, tuple):
            msg = " Note: " + new_key[1]
            new_key = new_key[0]
        else:
            msg = ""
        raise KeyError(
            "Key {} was renamed to {}; please update your config.{}".format(
                full_key, new_key, msg
            )
        )

    @classmethod
    def _load_dict_from_file(cls, cfg_filename):
        if cfg_filename.endswith(".ini"):
            config = configparser.ConfigParser()
            config.read(PathManager.get_local_path(cfg_filename))
            vars_dict = config['DEFAULT']
            vars_dict = {k.upper(): v for k, v in vars_dict}
        elif cfg_filename.endswith(".yaml") or cfg_filename.endswith(".yml"):
            with open(cfg_filename, "r") as f:
                vars_dict = yaml.load(f)
        else:
            raise ValueError("file {} not supported".format(cfg_filename))

        return vars_dict

    def set_attr(self, attr, val):
        """
        Wrapper method to set attributes of config
        :param attr: a string attr
        :param val: value to set attribute to

        :raises ValueError: if attr is not a string
                            if attr does not exist for the config
                            if type of val is different from the default type
        """
        if type(attr) is not str:
            raise ValueError('attr must be of type str')

        if not hasattr(self, attr):
            raise ValueError('The attribute %s does not exist' % attr)
        curr_val = self.__getattribute__(attr)

        if type(val) is str and type(curr_val) is not str:
            val = utils.convert_data_type(var_string=val, data_type=type(curr_val))

        if curr_val is not None and (type(val) != type(curr_val)):
            raise ValueError('%s is of type %s. Expected %s' % (attr, str(type(val)), str(type(curr_val))))

        self.__setattr__(attr, val)

    def change_to_test(self):
        """
        Initialize testing state
        """
        self.STATE = 'testing'

        # if cross validation is enabled, load testing cross validation bin
        if self.USE_CROSS_VALIDATION:
            assert self.CV_FILE, "No cross-validation file found in config"
            cv_processor = cv_util.CrossValidationProcessor(self.CV_FILE)
            bins = (self.CV_TRAIN_BINS, self.CV_VALID_BINS, self.CV_TEST_BINS)

            train_files, valid_files, test_files = cv_processor.get_fnames(bins)

            self.__CV_TRAIN_FILES__ = train_files
            self.__CV_VALID_FILES__ = valid_files
            self.__CV_TEST_FILES__ = test_files

    def summary(self, additional_vars=[]):
        """
        Print config summary
        :param additional_vars: additional list of variables to print
        :return:
        """

        summary_vals = ['MODEL_NAME', 'TAG', '']

        if self.STATE == 'training':
            summary_vals.extend([
                'TRAIN_DATASET', 'VAL_DATASET', 'TEST_DATASET', '',

                'CATEGORIES', '',

                'IMG_SIZE', '',

                'N_EPOCHS',
                'AUGMENT_DATA',
                'LOSS',
                "CLASS_WEIGHTS",
                "",

                'USE_CROSS_VALIDATION',
                'CV_K' if self.USE_CROSS_VALIDATION else '',
                'CV_FILE' if self.USE_CROSS_VALIDATION else '',
                'CV_TRAIN_BINS' if self.USE_CROSS_VALIDATION else '',
                'CV_VALID_BINS' if self.USE_CROSS_VALIDATION else '',
                'CV_TEST_BINS' if self.USE_CROSS_VALIDATION else '', ''
                                                                     
                'TRAIN_BATCH_SIZE', 'VALID_BATCH_SIZE', '',

                "NUM_GRAD_STEPS", "",

                'INITIAL_LEARNING_RATE',
                'LR_SCHEDULER_NAME',
                'DROP_FACTOR' if self.LR_SCHEDULER_NAME else '',
                'DROP_RATE' if self.LR_SCHEDULER_NAME else '',
                'MIN_LEARNING_RATE' if self.LR_SCHEDULER_NAME else '',
                "LR_MIN_DELTA" if self.LR_SCHEDULER_NAME else "",
                "LR_PATIENCE" if self.LR_SCHEDULER_NAME else "",
                "LR_COOLDOWN" if self.LR_SCHEDULER_NAME else "",
                ""

                'USE_EARLY_STOPPING',
                'EARLY_STOPPING_MIN_DELTA' if self.USE_EARLY_STOPPING else '',
                'EARLY_STOPPING_PATIENCE' if self.USE_EARLY_STOPPING else '',
                'EARLY_STOPPING_CRITERION' if self.USE_EARLY_STOPPING else '',
                '',

                'KERNEL_INITIALIZER',
                'SEED' if self.SEED else '', '' 

                'FINE_TUNE',
                'INIT_WEIGHTS', '',

                'NUM_WORKERS',
                "OUTPUT_DIR",
                '',
            ])
        else:
            summary_vals.extend([
                'TEST_RESULT_PATH',
                'TEST_WEIGHT_PATH',
                'TEST_BATCH_SIZE',
            ])

        summary_vals.extend(additional_vars)

        # Remove consecutive elements in summary vals that are the same
        summary_vals = [x[0] for x in groupby(summary_vals)]

        logger.info('')
        logger.info('==' * 40)
        logger.info("Config Summary")
        logger.info('==' * 40)

        for attr in summary_vals:
            if attr == '':
                logger.info('')
                continue
            logger.info(attr + ": " + str(self.__getattribute__(attr)))

        logger.info('==' * 40)
        logger.info('')

    def get_num_classes(self):
        if self.INCLUDE_BACKGROUND:
            return len(self.CATEGORIES) + 1

        return len(self.CATEGORIES)

    def num_neighboring_slices(self):
        return None

    @property
    def testing(self):
        return self.STATE == 'testing'

    @property
    def training(self):
        return self.STATE == 'training'

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subcommand_parser = parser.add_parser('%s' % cls.MODEL_NAME, description='%s config parameters')

        # Data format tag
        subcommand_parser.add_argument('--tag', metavar='T', type=str, default=cls.TAG, nargs='?',
                                       help='tag defining data format. Default: %s' % cls.TAG)

        # Data paths
        subcommand_parser.add_argument('--train_dataset', metavar='tp', type=str, default=cls.TRAIN_DATASET, nargs='?',
                                       help='training data path. Default: %s' % cls.TRAIN_DATASET)
        subcommand_parser.add_argument('--val_dataset', metavar='tp', type=str, default=cls.VAL_DATASET, nargs='?',
                                       help='validation data path. Default: %s' % cls.VAL_DATASET)
        subcommand_parser.add_argument('--test_dataset', metavar='tp', type=str, default=cls.TEST_DATASET, nargs='?',
                                       help='testing data path. Default: %s' % cls.TEST_DATASET)

        # Number of epochs
        subcommand_parser.add_argument('--n_epochs', metavar='E', type=int, default=cls.N_EPOCHS, nargs='?',
                                       help='number of training epochs. Default: %d' % cls.N_EPOCHS)

        # Augment data
        subcommand_parser.add_argument('--augment_data', default=False, action='store_const',
                                       const=True,
                                       help='use augmented data for training. Default: %s' % False)

        # Learning rate.
        subcommand_parser.add_argument('--initial_learning_rate', metavar='LR', type=float,
                                       default=cls.INITIAL_LEARNING_RATE,
                                       nargs='?',
                                       help='initial learning rate. Default: %s' % cls.INITIAL_LEARNING_RATE)
        subcommand_parser.add_argument('--min_learning_rate', metavar='mLR', type=float,
                                       default=cls.MIN_LEARNING_RATE,
                                       nargs='?',
                                       help='minimum learning rate during decay. Default: %s' % cls.MIN_LEARNING_RATE)
        subcommand_parser.add_argument(
            '--lr_scheduler_name',
            default=cls.LR_SCHEDULER_NAME,
            nargs="?",
            type=str,
            help='learning rate scheduler. Default: {}'.format(
                None if not cls.LR_SCHEDULER_NAME else cls.LR_SCHEDULER_NAME),
        )
        subcommand_parser.add_argument('--drop_factor', metavar='DF', type=float, default=cls.DROP_FACTOR, nargs='?',
                                       help='drop factor for learning rate decay. Default: %s' % cls.DROP_FACTOR)
        subcommand_parser.add_argument('--drop_rate', metavar='DR', type=float, default=cls.DROP_RATE, nargs='?',
                                       help='drop rate for learning rate decay. Default: %s' % cls.DROP_RATE)

        # Number of gradient steps
        subcommand_parser.add_argument('--num_grad_steps', metavar='S', type=int,
                               default=cls.NUM_GRAD_STEPS,
                               nargs='?',
                               help='number of gradient accumulation steps. Default: %s' % cls.NUM_GRAD_STEPS)

        # Early stopping
        subcommand_parser.add_argument('--use_early_stopping', default=False, action='store_const',
                                       const=True,
                                       help='use learning rate step decay. Default: %s' % False)
        subcommand_parser.add_argument('--early_stopping_min_delta', metavar='D', type=float,
                                       default=cls.EARLY_STOPPING_MIN_DELTA, nargs='?',
                                       help='minimum change in the monitored quantity to qualify as an improvement, '
                                            'i.e. an absolute change of less than min_delta, will count as no improvement. Default: %s' % cls.EARLY_STOPPING_MIN_DELTA)
        subcommand_parser.add_argument('--early_stopping_patience', metavar='P', type=int, default=0, nargs='?',
                                       help='number of epochs with no improvement after which training will be stopped. Default: %s' % cls.EARLY_STOPPING_PATIENCE)
        subcommand_parser.add_argument('--early_stopping_criterion', metavar='C', type=str, default='val_loss',
                                       nargs='?',
                                       help='criterion to monitor for early stopping. Default: %s' % cls.EARLY_STOPPING_CRITERION)

        # Batch size
        subcommand_parser.add_argument('--train_batch_size', metavar='trBS', type=int, default=cls.TRAIN_BATCH_SIZE,
                                       nargs='?',
                                       help='training batch size. Default: %s' % cls.TRAIN_BATCH_SIZE)
        subcommand_parser.add_argument('--valid_batch_size', metavar='vBS', type=int, default=cls.VALID_BATCH_SIZE,
                                       nargs='?',
                                       help='validation batch size. Default: %s' % cls.VALID_BATCH_SIZE)
        subcommand_parser.add_argument('--test_batch_size', metavar='tBS', type=int, default=cls.TEST_BATCH_SIZE,
                                       nargs='?',
                                       help='testing/inference batch size. Default %s' % cls.TEST_BATCH_SIZE)

        # Loss function
        subcommand_parser.add_argument('--loss', metavar='L', type=str, default='DICE_LOSS', nargs='?',
                                       choices=CMD_LINE_SUPPORTED_LOSSES,
                                       help='loss function. Choose from %s' % CMD_LINE_SUPPORTED_LOSSES)
        subcommand_parser.add_argument('--class_weights', metavar="W", type=float, default=cls.CLASS_WEIGHTS,
                                       nargs="*",
                                       help="class weights (if applicable). Defaults to equal weighting",
                                      )

        # Include background
        subcommand_parser.add_argument('--include_background', default=False, action='store_const',
                                       const=True,
                                       help='include background for loss function (i.e. softmax). Default: %s' % False)

        # Image size
        subcommand_parser.add_argument('--img_size', type=str, default=str(cls.IMG_SIZE), nargs='?',
                                       help='image size. Default: %s' % str(cls.IMG_SIZE))

        # Kernel initializer
        subcommand_parser.add_argument('--kernel_initializer', type=str, default=cls.KERNEL_INITIALIZER, nargs='?',
                                       help='kernel initializer. Default: %s' % str(cls.KERNEL_INITIALIZER))

        subcommand_parser.add_argument('-s', '--seed', metavar='S', type=int, default=cls.SEED, nargs='?',
                                       dest='seed',
                                       help='python seed to initialize filter weights. Default: %s' % cls.SEED)

        # Initialize weight path.
        subcommand_parser.add_argument('-init_weight_path', '--init_weight_path', metavar='P', type=str,
                                       default=cls.INIT_WEIGHTS,
                                       nargs='?',
                                       dest='init_weight_path',
                                       help='Path to weights file to initialize. Default: %s' % cls.INIT_WEIGHTS)

        # System parameters
        subcommand_parser.add_argument('--num_workers', metavar='W', type=int, default=1, nargs='?',
                                       dest='num_workers',
                                       help='number of workers for data loading. Default: %s' % cls.NUM_WORKERS)

        return subcommand_parser

    @classmethod
    def __get_cmd_line_vars__(cls):
        return ['tag',
                'train_dataset', 'val_dataset', 'test_dataset',
                'n_epochs', 'augment_data',
                'num_grad_steps',
                'lr_scheduler_name', 'initial_learning_rate',
                'min_learning_rate', 'drop_factor', 'drop_rate',
                'use_early_stopping', 'early_stopping_min_delta',
                'early_stopping_patience',
                'early_stopping_criterion',
                'train_batch_size', 'valid_batch_size', 'test_batch_size',
                'loss', 'class_weights', 'include_background',
                'img_size',
                'kernel_initializer', 'seed',
                'init_weight_path',
                'num_workers',
                ]

    @classmethod
    def parse_cmd_line(cls, vargin):
        config_dict = dict()
        for skey in cls.__get_cmd_line_vars__():
            if skey not in vargin.keys():
                continue

            c_skey = skey.upper()
            val = vargin[skey]

            if skey == 'loss':
                val = get_training_loss_from_str(vargin[skey].upper())

            if skey == 'img_size':
                val = utils.convert_data_type(vargin[skey], data_type=type(cls.IMG_SIZE))
                assert type(val) is tuple

            config_dict[c_skey] = val
        return config_dict


class DeeplabV3Config(Config):
    """
    Configuration for 2D Deeplabv3+ architecture (https://arxiv.org/abs/1802.02611)
    """
    MODEL_NAME = "deeplabv3_2d"

    OS = 16
    DIL_RATES = (2, 4, 6)
    AT_DIVISOR = 2
    DROPOUT_RATE = 0.1

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    def summary(self, additional_vars=[]):
        summary_attrs = ['OS', 'DIL_RATES', 'DROPOUT_RATE']
        super().summary(summary_attrs)

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subparser = super().init_cmd_line_parser(parser)

        subparser.add_argument('--os', type=int, default=cls.OS, nargs='?',
                               help='output stride. Default: %d' % cls.OS)
        subparser.add_argument('--dil_rates', type=str, default=str(cls.DIL_RATES), nargs='?',
                               help='dilation rates. Default: %s' % str(cls.DIL_RATES))
        subparser.add_argument('--dropout_rate', type=float, default=cls.DROPOUT_RATE, nargs='?',
                               help='dropout rate before classification layer')

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend(['os', 'dil_rates', 'dropout_rate'])
        return cmd_line_vars

    @classmethod
    def parse_cmd_line(cls, vargin) -> dict:
        config_dict = super().parse_cmd_line(vargin)
        
        config_dict['DIL_RATES'] = utils.convert_data_type(config_dict['DIL_RATES'], tuple)

        assert len(config_dict['DIL_RATES']) == 3

        return config_dict


class SegnetConfig(Config):
    """
    Configuration for 2D Segnet architecture (https://arxiv.org/abs/1505.07293)
    """
    MODEL_NAME = "segnet_2d"

    TRAIN_BATCH_SIZE = 15

    DEPTH = 6
    NUM_CONV_LAYERS = [2, 2, 3, 3, 3, 3]
    NUM_FILTERS = [64, 128, 256, 256, 512, 512]

    SINGLE_BN = False
    CONV_ACT_BN = False
    USE_BOTTLENECK = False

    INITIAL_LEARNING_RATE = 1e-3

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    def summary(self, additional_vars=[]):
        summary_attrs = ['DEPTH', 'NUM_CONV_LAYERS', 'NUM_FILTERS']
        super().summary(summary_attrs)

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subparser = super().init_cmd_line_parser(parser)

        subparser.add_argument('--depth', type=int, default=cls.DEPTH, nargs='?',
                               help='network depth. Default: %d' % cls.DEPTH)
        subparser.add_argument('--num_conv_layers', type=str, default=str(cls.NUM_CONV_LAYERS), nargs='?',
                               help='number of convolutional layers. Default: %s' % str(cls.NUM_CONV_LAYERS))
        subparser.add_argument('--num_filters', type=str, default=str(cls.NUM_FILTERS), nargs='?',
                               help='number of filters at each depth layer. Default: %s' % str(cls.NUM_FILTERS))
        subparser.add_argument(
            "--single_bn",
            default=False,
            action="store_true",
            help='use single batch norm per depth. Default: %s' % False,
        )
        subparser.add_argument(
            "--use_bottleneck",
            default=False,
            action="store_true",
            help='use bottleneck w/o pooling. Default: %s' % False,
        )

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend([
            'depth',
            'num_conv_layers',
            'num_filters',
            "single_bn",
            "use_bottleneck",
        ])
        return cmd_line_vars

    @classmethod
    def parse_cmd_line(cls, vargin) -> dict:
        config_dict = super().parse_cmd_line(vargin)
        depth = config_dict['DEPTH']

        num_conv_layers = utils.convert_data_type(config_dict['NUM_CONV_LAYERS'], type(cls.NUM_CONV_LAYERS))
        num_filters = utils.convert_data_type(config_dict['NUM_FILTERS'], type(cls.NUM_FILTERS))

        assert len(num_conv_layers) == depth, "Number of conv layers must be specified for each depth"
        assert len(num_filters) == depth, "Number of filters must be specified for each depth"

        config_dict['NUM_CONV_LAYERS'] = num_conv_layers
        config_dict['NUM_FILTERS'] = num_filters

        return config_dict


class UNetConfig(Config):
    """
    Configuration for 2D U-Net architecture (https://arxiv.org/abs/1505.04597)
    """
    MODEL_NAME = "unet_2d"

    INIT_UNET_2D = False

    INITIAL_LEARNING_RATE = 2e-2
    DROP_FACTOR = 0.8 ** (1 / 5)
    DROP_RATE = 1.0
    TRAIN_BATCH_SIZE = 35

    DEPTH = 6
    NUM_FILTERS = None

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subparser = super().init_cmd_line_parser(parser)

        subparser.add_argument('--depth', type=int, default=cls.DEPTH, nargs='?',
                               help='network depth. Default: %d' % cls.DEPTH)
        subparser.add_argument('--num_filters', type=str, default=str(cls.NUM_FILTERS), nargs='?',
                               help='number of filters. Default: %s' % str(cls.NUM_FILTERS))

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend(['depth', 'num_filters'])
        return cmd_line_vars

    def summary(self, additional_vars=[]):
        summary_vars = ['DEPTH', 'NUM_FILTERS', '']
        summary_vars.extend(additional_vars)
        super().summary(summary_vars)

    @classmethod
    def parse_cmd_line(cls, vargin) -> dict:
        config_dict = super().parse_cmd_line(vargin)
        depth = config_dict['DEPTH']

        num_filters = utils.convert_data_type(config_dict['NUM_FILTERS'], type(cls.NUM_FILTERS))
        if num_filters:
            assert len(num_filters) == depth, "Number of filters must be specified for each depth"

        config_dict['NUM_FILTERS'] = num_filters

        return config_dict

class ResidualUNet(Config):
    """
    Configuration for 2D Residual U-Net architecture
    """
    MODEL_NAME = 'res_unet'

    DEPTH = 6
    NUM_FILTERS = None

    DROPOUT_RATE = 0.0
    LAYER_ORDER = ['relu', 'bn', 'dropout', 'conv']

    USE_SE_BLOCK = False
    SE_RATIO = 8

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subparser = super().init_cmd_line_parser(parser)

        subparser.add_argument('--depth', type=int, default=cls.DEPTH, nargs='?',
                               help='network depth. Default: %d' % cls.DEPTH)
        subparser.add_argument('--dropout_rate', type=float, default=cls.DROPOUT_RATE, nargs='?',
                               help='dropout rate. Default: %d' % cls.DROPOUT_RATE)
        subparser.add_argument('--layer_order', type=str, default=str(cls.LAYER_ORDER), nargs='?',
                               help='layer order. Default: %s' % cls.LAYER_ORDER)

        subparser.add_argument('--use_se_block', action='store_const', default=False, const=True,
                               help='use squeeze-excitation block. Default: False')
        subparser.add_argument('--se_ratio', type=int, default=cls.SE_RATIO, nargs='?',
                               help='squeeze-excitation downsampling ratio. Default: %d' % cls.SE_RATIO)

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend(['depth', 'dropout_rate', 'layer_order', 'use_se_block', 'se_ratio'])
        return cmd_line_vars

    def summary(self, additional_vars=[]):
        summary_attrs = ['DEPTH', 'NUM_FILTERS', 'DROPOUT_RATE', '',
                         'LAYER_ORDER', '',
                         'USE_SE_BLOCK', 'SE_RATIO']
        super().summary(summary_attrs)

    def num_neighboring_slices(self):
        return self.IMG_SIZE[-1] if self.IMG_SIZE[-1] != 1 else None


class EnsembleUDSConfig(Config):
    MODEL_NAME = "ensemble_uds"
    N_EPOCHS = 100

    def __init__(self, state='training', create_dirs=True):
        raise DeprecationWarning('This config is deprecated')
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class UNetMultiContrastConfig(UNetConfig):
    IMG_SIZE = (288, 288, 3)

    MODEL_NAME = 'unet_2d_multi_contrast'

    # Whether to load weights from original unet model
    # INIT_UNET_2D = True
    # INIT_UNET_2D_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'

    def __init__(self, state='training', create_dirs=True):
        super().__init__(state, create_dirs=create_dirs)


class UNet2_5DConfig(UNetConfig):
    """
    Configuration for 3D U-Net architecture
    """

    IMG_SIZE = (288, 288, 7)

    MODEL_NAME = 'unet_2_5d'
    N_EPOCHS = 20
    AUGMENT_DATA = False
    INITIAL_LEARNING_RATE = 1e-2

    DROP_RATE = 1.0
    DROP_FACTOR = 0.8

    # Train path - volumetric augmentation
    #TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


class UNet3DConfig(UNetConfig):

    IMG_SIZE = (288, 288, 4, 1)

    MODEL_NAME = 'unet_3d'
    N_EPOCHS = 20
    INITIAL_LEARNING_RATE = 1e-2

    DROP_RATE = 1.0
    DROP_FACTOR = 0.8

    TAG = 'oai_3d'

    SLICE_SUBSET = None  # 1 indexed inclusive - i.e. (5, 64) means slices [5, 64]

    NUM_FILTERS = [32, 64, 128, 256, 512, 1024]

    # Train path - volumetric augmentation
    #TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subparser = super().init_cmd_line_parser(parser)

        subparser.add_argument('--slice_subset', type=str, default=str(cls.SLICE_SUBSET), nargs='?',
                               help='subset of slices to select (tuple). Default: %s' % str(cls.SLICE_SUBSET))

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend(['slice_subset'])
        return cmd_line_vars

    def summary(self, additional_vars=[]):
        summary_attrs = ['SLICE_SUBSET']
        super().summary(summary_attrs)

    @classmethod
    def parse_cmd_line(cls, vargin) -> dict:
        config_dict = super().parse_cmd_line(vargin)

        slice_subset = utils.convert_data_type(config_dict['SLICE_SUBSET'], type(cls.SLICE_SUBSET))
        if slice_subset:
            assert len(slice_subset) == 2, "slice_subset must define starting and ending slices"

        config_dict['SLICE_SUBSET'] = slice_subset

        return config_dict


class DeeplabV3_2_5DConfig(DeeplabV3Config):
    """
    Configuration for 2.5D Deeplabv3+ architecture
    """
    IMG_SIZE = (288, 288, 3)

    MODEL_NAME = 'deeplabv3_2_5d'
    N_EPOCHS = 100

    # Train path - volumetric augmentation
    #TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


class AnisotropicUNetConfig(Config):
    """
    Configuration for 2D Anisotropic U-Net architecture
    """
    MODEL_NAME = 'anisotropic_unet'

    IMG_SIZE = (288, 72, 1)

    INITIAL_LEARNING_RATE = 2e-2
    DROP_FACTOR = 0.85
    DROP_RATE = 1.0
    TRAIN_BATCH_SIZE = 60

    DEPTH = 6
    NUM_FILTERS = None

    KERNEL_SIZE = (7, 3)

    #KERNEL_SIZE_RATIO = None
    #POOLING_SIZE_RATIO = None
    #POOLING_SIZE = (3, 11)

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    @classmethod
    def init_cmd_line_parser(cls, parser):
        subparser = super().init_cmd_line_parser(parser)

        subparser.add_argument('--depth', type=int, default=cls.DEPTH, nargs='?',
                               help='network depth. Default: %d' % cls.DEPTH)
        subparser.add_argument('--kernel_size', type=str, default=str(cls.KERNEL_SIZE), nargs='?',
                               help='kernel size. Default: %s' % str(cls.KERNEL_SIZE))

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend(['depth', 'kernel_size'])
        return cmd_line_vars

    def summary(self, additional_vars=[]):
        summary_attrs = ['DEPTH', 'NUM_FILTERS', 'KERNEL_SIZE']
        super().summary(summary_attrs)


class RefineNetConfig(Config):
    """
    Configuration for RefineNet architecture as suggested by paper below
    http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf
    """
    MODEL_NAME = 'refinenet'

    INITIAL_LEARNING_RATE = 1e-3

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


def _check_and_coerce_cfg_value_type(replacement, original, full_key):
    """Checks that `replacement`, which is intended to replace `original` is of
    the right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    original_type = type(original)
    replacement_type = type(replacement)

    # TODO: Convert all to have non-None values by default.
    if original_type == type(None):
        return replacement

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    # Cast replacement from from_type to to_type if the replacement and original
    # types match from_type and to_type
    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    # Conditionally casts
    # list <-> tuple
    casts = [(tuple, list), (list, tuple)]

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )


def _assert_with_logging(cond, msg):
    if not cond:
        logger.debug(msg)
    assert cond, msg


def _error_with_logging(cond, msg, error_type=ValueError):
    if not cond:
        logger.error(msg)
        raise error_type(msg)


SUPPORTED_CONFIGS = [UNetConfig, SegnetConfig, DeeplabV3Config, ResidualUNet,
                     AnisotropicUNetConfig, RefineNetConfig,
                     UNet3DConfig, UNet2_5DConfig, DeeplabV3_2_5DConfig]


def get_config(
    config_cp_save_tag: str,
    create_dirs: bool=True,
    output_dir: str = "",
):
    """Get config using config_cp_save_tag

    Args:
        config_cp_save_tag: config cp_save_tag
        create_dirs: if directory should be created

    Return:
        Config: A config.
    """

    configs = SUPPORTED_CONFIGS
    for config in configs:
        if config.CP_SAVE_TAG == config_cp_save_tag:
            c = config(create_dirs=create_dirs)
            if output_dir:
                c.OUTPUT_DIR = output_dir
            return c

    raise ValueError('config %s not found' % config_cp_save_tag)


def get_model_name(cfg_filename: str):
    """Get "MODEL_NAME" from config file.
    Args:
        cfg_filename: filepath to INI or YAML file where config is stored

    Returns:
        str: MODEL_NAME
    """
    vars_dict = Config._load_dict_from_file(cfg_filename)
    return vars_dict['MODEL_NAME']


def init_cmd_line_parser(parser):
    """
    Initialize command line parser for configs by adding supported configs as arguments
    :param parser: an ArgumentParser
    :return: subparsers corresponding to command line arguments for each config
    """
    subparsers = []
    for config in SUPPORTED_CONFIGS:
        subparsers.append(config.init_cmd_line_parser(parser))
    return subparsers


def config_exists(experiment_dir: str):
    return os.path.isfile(os.path.join(experiment_dir, "config.ini")) \
           or os.path.isfile(os.path.join(experiment_dir, "config.yaml")) \
           or os.path.isfile(os.path.join(experiment_dir, "config.yml"))
