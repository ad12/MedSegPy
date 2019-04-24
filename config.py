import configparser
import os
import warnings
from itertools import groupby
from time import localtime, strftime

import glob_constants as glc
import mri_utils
import utils.utils as utils
from cross_validation import cv_utils
from losses import DICE_LOSS, CMD_LINE_SUPPORTED_LOSSES, get_training_loss_from_str
from utils import io_utils

DEPRECATED_KEYS = ['NUM_CLASSES', 'TRAIN_FILES_CV', 'VALID_FILES_CV', 'TEST_FILES_CV']

DEEPLABV3_NAME = 'deeplabv3_2d'
SEGNET_NAME = 'segnet_2d'
UNET_NAME = 'unet_2d'
ENSEMBLE_UDS_NAME = 'ensemble_uds'

# This is the default save path prefix - please change if you desire something else
SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/oai_data'

SUPPORTED_CONFIGS_NAMES = [DEEPLABV3_NAME, SEGNET_NAME, UNET_NAME]


class Config():
    VERSION = 5

    # Loss function in form (id, output_mode)
    LOSS = DICE_LOSS

    # PIDS to include, None = all pids
    PIDS = None

    DEBUG = False

    # Model architecture path
    PLOT_MODEL_PATH = io_utils.check_dir('./model_imgs')

    # Training and validation image size
    IMG_SIZE = (288, 288, 1)

    # Training parameters
    N_EPOCHS = 100
    AUGMENT_DATA = False
    USE_STEP_DECAY = False

    # Step Decay params
    INITIAL_LEARNING_RATE = 1e-4
    MIN_LEARNING_RATE = 1e-8
    DROP_FACTOR = 0.7
    DROP_RATE = 1.0

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

    # Tissues to render
    TISSUES = [mri_utils.MASK_FEMORAL_CARTILAGE]
    INCLUDE_BACKGROUND = False

    # File Types
    FILE_TYPES = ['im']

    # Transfer Learning
    FINE_TUNE = False
    INIT_WEIGHT_PATH = ''

    # Dataset Paths
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/'
    VALID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/'
    TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'

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

    # test result folder name
    TEST_RESULTS_FOLDER_NAME = 'test_results'

    # Training Model Paths
    CP_SAVE_TAG = ''
    CP_SAVE_PATH = ''
    PIK_SAVE_PATH = ''
    PIK_SAVE_PATH_DIR = ''
    TF_LOG_DIR = ''

    # Test Result Path
    TEST_RESULT_PATH = ''
    TEST_WEIGHT_PATH = ''

    # Dataset tag - What dataset are we training on? 'dess' or 'oai'
    # choose from oai_aug, oai_aug_3d
    TAG = 'oai_aug'

    # Restrict number of files learned. Default is all []
    LEARN_FILES = []

    # Initializer
    KERNEL_INITIALIZER = 'he_normal'

    def __init__(self, cp_save_tag, state='training', create_dirs=True):
        self.SEED = glc.SEED
        if state not in ['testing', 'training']:
            raise ValueError('state must either be \'training\' or \'testing\'')

        self.DATE_TIME_STR = strftime("%Y-%m-%d-%H-%M-%S", localtime())

        inference = state == 'testing'
        self.CP_SAVE_TAG = cp_save_tag
        self.STATE = state

        if create_dirs:
            if not inference:
                # training
                prefix = self.DATE_TIME_STR
                self.init_training_paths(prefix)
            else:
                # Testing
                self.TEST_RESULT_PATH = io_utils.check_dir(
                    os.path.join('./' + self.TEST_RESULTS_FOLDER_NAME, self.CP_SAVE_TAG, self.TAG, self.DATE_TIME_STR))

    def init_training_paths(self, prefix):
        """
        Intitialize training paths
        :param prefix: a string to uniquely identify this experiment
        """
        self.CP_SAVE_PATH = io_utils.check_dir(os.path.join(SAVE_PATH_PREFIX, self.CP_SAVE_TAG, prefix))
        self.PIK_SAVE_PATH = os.path.join(self.CP_SAVE_PATH, 'pik_data.dat')
        self.PIK_SAVE_PATH_DIR = io_utils.check_dir(os.path.dirname(self.PIK_SAVE_PATH))
        self.TF_LOG_DIR = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH, 'tf_log'))

    def init_fine_tune(self, init_weight_path):
        """
        Initialize fine tune state
        :param init_weight_path: path to initial weights
        """
        if (self.STATE != 'training'):
            raise ValueError('Must be in training state')

        self.FINE_TUNE = True
        self.INIT_WEIGHT_PATH = init_weight_path

        prefix = os.path.join(self.DATE_TIME_STR, 'fine_tune')

        # if fine_tune folder already exists, do not overwrite it
        count = 2
        while os.path.isdir(os.path.join(SAVE_PATH_PREFIX, self.CP_SAVE_TAG, prefix)):
            prefix = os.path.join(self.DATE_TIME_STR, 'fine_tune_%03d' % count)
            count += 1

        self.init_training_paths(prefix)

    def init_cross_validation(self, bins_files,
                              train_bins, valid_bins, test_bins,
                              cv_k, cv_file, cp_save_path):
        assert self.STATE == 'training', "To initialize cross-validation, must be in training state"

        self.USE_CROSS_VALIDATION = True
        self.CV_TRAIN_BINS = train_bins
        self.CV_VALID_BINS = valid_bins
        self.CV_TEST_BINS = test_bins
        self.CV_K = cv_k
        self.CV_FILE = cv_file

        train_files, valid_files, test_files = cv_utils.get_fnames(bins_files, (train_bins, valid_bins, test_bins))
        self.__CV_TRAIN_FILES__ = train_files
        self.__CV_VALID_FILES__ = valid_files
        self.__CV_TEST_FILES__ = test_files

        self.CP_SAVE_PATH = io_utils.check_dir(cp_save_path)
        self.PIK_SAVE_PATH = os.path.join(self.CP_SAVE_PATH, 'pik_data.dat')
        self.PIK_SAVE_PATH_DIR = io_utils.check_dir(os.path.dirname(self.PIK_SAVE_PATH))
        self.TF_LOG_DIR = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH, 'tf_log'))

    def save_config(self):
        """
        Save params of config to ini file
        """
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__") and not (hasattr(type(self), attr) and isinstance(getattr(type(self), attr), property))]
        
        filepath = os.path.join(self.CP_SAVE_PATH, 'config.ini')

        config_vars = dict()
        for m_var in members:
            config_vars[m_var] = getattr(self, m_var)

        # Save config
        config = configparser.ConfigParser(config_vars)
        with open(filepath, 'w+') as configfile:
            config.write(configfile)

        # Save as object to make it easy to load
        filepath = os.path.join(self.CP_SAVE_PATH, 'config_obj.dat')
        io_utils.save_pik(self, filepath)

    def load_config(self, ini_filepath):
        """
        Load params of config from ini file
        :param ini_filepath: path to ini file
        """
        config = configparser.ConfigParser()
        config.read(ini_filepath)
        vars_dict = config['DEFAULT']
        
        if vars_dict['CP_SAVE_TAG'] != self.CP_SAVE_TAG:
            raise ValueError('Wrong config. Expected %s' % str(vars_dict['CP_SAVE_TAG']))

        for key in vars_dict.keys():
            upper_case_key = str(key).upper()
            
            if upper_case_key in DEPRECATED_KEYS:
                warnings.warn('Key %s is deprecated, not loading' % upper_case_key)
                continue

            if not hasattr(self, upper_case_key):
                raise ValueError(
                    'Key %s does not exist. Please make sure all variable names are fully capitalized' % upper_case_key)

            # all data is of type string, but we need to cast back to original data type
            data_type = type(getattr(self, upper_case_key))

            # print(upper_case_key + ': ' + str(vars_dict[key]) + ' (' + str(data_type) + ')')
            var_converted = utils.convert_data_type(vars_dict[key], data_type)

            # Loading config
            self.__setattr__(str(key).upper(), var_converted)

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
        self.TEST_RESULT_PATH = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH, self.TEST_RESULTS_FOLDER_NAME))

        # if cross validation is enabled, load testing cross validation bin
        if self.USE_CROSS_VALIDATION:
            train_files, valid_files, test_files = cv_utils.get_fnames(cv_utils.load_cross_validation(self.CV_K),
                                                                       (self.CV_TRAIN_BINS, self.CV_VALID_BINS,
                                                                        self.CV_TEST_BINS))
            self.__CV_TRAIN_FILES__ = train_files
            self.__CV_VALID_FILES__ = valid_files
            self.__CV_TEST_FILES__ = test_files

    def summary(self, additional_vars=[]):
        """
        Print config summary
        :param additional_vars: additional list of variables to print
        :return:
        """

        summary_vals = ['CP_SAVE_TAG', 'TAG', '']

        if self.STATE == 'training':
            summary_vals.extend([
                'TRAIN_PATH', 'VALID_PATH', 'TEST_PATH', '',

                'TISSUES', '',

                'IMG_SIZE', '',

                'N_EPOCHS', 'AUGMENT_DATA', 'LOSS', '',

                'USE_CROSS_VALIDATION', 'CV_K' if self.USE_CROSS_VALIDATION else '',
                'CV_TRAIN_BINS' if self.USE_CROSS_VALIDATION else '',
                'CV_VALID_BINS' if self.USE_CROSS_VALIDATION else '',
                'CV_TEST_BINS' if self.USE_CROSS_VALIDATION else '', ''
                                                                     
                'TRAIN_BATCH_SIZE', 'VALID_BATCH_SIZE', '',

                'INITIAL_LEARNING_RATE',
                'USE_STEP_DECAY',
                'DROP_FACTOR' if self.USE_STEP_DECAY else '',
                'DROP_RATE' if self.USE_STEP_DECAY else '',
                'MIN_LEARNING_RATE' if self.USE_STEP_DECAY else '', '',

                'USE_EARLY_STOPPING',
                'EARLY_STOPPING_MIN_DELTA' if self.USE_EARLY_STOPPING else '',
                'EARLY_STOPPING_PATIENCE' if self.USE_EARLY_STOPPING else '',
                'EARLY_STOPPING_CRITERION' if self.USE_EARLY_STOPPING else '', '',

                'KERNEL_INITIALIZER', '',

                'FINE_TUNE',
                'INIT_WEIGHT_PATH' if self.FINE_TUNE else ''])
        else:
            summary_vals.extend(['TEST_RESULT_PATH', 'TEST_WEIGHT_PATH', 'TEST_BATCH_SIZE'])

        summary_vals.extend(additional_vars)

        # Remove consecutive elements in summary vals that are the same
        summary_vals = [x[0] for x in groupby(summary_vals)]

        print('')
        print('==' * 40)
        print("Config Summary")
        print('==' * 40)

        for attr in summary_vals:
            if attr == '':
                print('')
                continue
            print(attr + ": " + str(self.__getattribute__(attr)))

        print('==' * 40)
        print('')

    def get_num_classes(self):
        if self.INCLUDE_BACKGROUND:
            return len(self.TISSUES) + 1

        return len(self.TISSUES)

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
        subcommand_parser = parser.add_parser('%s' % cls.CP_SAVE_TAG, description='%s config parameters')

        # Data format tag
        subcommand_parser.add_argument('--tag', metavar='T', type=str, default=cls.TAG, nargs='?',
                                       help='tag defining data format. Default: %s' % cls.TAG)

        # Data paths
        subcommand_parser.add_argument('--train_path', metavar='tp', type=str, default=cls.TRAIN_PATH, nargs='?',
                                       help='training data path. Default: %s' % cls.TRAIN_PATH)
        subcommand_parser.add_argument('--valid_path', metavar='tp', type=str, default=cls.VALID_PATH, nargs='?',
                                       help='validation data path. Default: %s' % cls.VALID_PATH)
        subcommand_parser.add_argument('--test_path', metavar='tp', type=str, default=cls.TEST_PATH, nargs='?',
                                       help='testing data path. Default: %s' % cls.TEST_PATH)

        # Number of epochs
        subcommand_parser.add_argument('--n_epochs', metavar='E', type=int, default=cls.N_EPOCHS, nargs='?',
                                       help='number of training epochs. Default: %d' % cls.N_EPOCHS)

        # Augment data
        subcommand_parser.add_argument('--augment_data', default=False, action='store_const',
                                       const=True,
                                       help='use augmented data for training. Default: %s' % False)

        # Learning rate step decay
        subcommand_parser.add_argument('--use_step_decay', default=False, action='store_const',
                                       const=True,
                                       help='use learning rate step decay. Default: %s' % False)
        subcommand_parser.add_argument('--initial_learning_rate', metavar='LR', type=float,
                                       default=cls.INITIAL_LEARNING_RATE,
                                       nargs='?',
                                       help='initial learning rate. Default: %s' % cls.INITIAL_LEARNING_RATE)
        subcommand_parser.add_argument('--min_learning_rate', metavar='mLR', type=float,
                                       default=cls.MIN_LEARNING_RATE,
                                       nargs='?',
                                       help='minimum learning rate during decay. Default: %s' % cls.MIN_LEARNING_RATE)
        subcommand_parser.add_argument('--drop_factor', metavar='DF', type=float, default=cls.DROP_FACTOR, nargs='?',
                                       help='drop factor for learning rate decay. Default: %s' % cls.DROP_FACTOR)
        subcommand_parser.add_argument('--drop_rate', metavar='DR', type=float, default=cls.DROP_RATE, nargs='?',
                                       help='drop rate for learning rate decay. Default: %s' % cls.DROP_RATE)

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

        return subcommand_parser

    @classmethod
    def __get_cmd_line_vars__(cls):
        return ['tag',
                'train_path', 'valid_path', 'test_path',
                'n_epochs', 'augment_data',
                'use_step_decay', 'initial_learning_rate', 'min_learning_rate', 'drop_factor', 'drop_rate',
                'use_early_stopping', 'early_stopping_min_delta', 'early_stopping_patience',
                'early_stopping_criterion',
                'train_batch_size', 'valid_batch_size', 'test_batch_size',
                'loss', 'include_background',
                'img_size']

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
    CP_SAVE_TAG = DEEPLABV3_NAME

    OS = 16
    DIL_RATES = (2, 4, 6)
    AT_DIVISOR = 2
    DROPOUT_RATE = 0.1

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    def change_to_test(self):
        self.state = 'testing'
        config_tuple = (self.OS,) + self.DIL_RATES
        config_str = '%d_%d-%d-%d' % config_tuple
        self.TEST_RESULT_PATH = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH,
                                                                self.TEST_RESULTS_FOLDER_NAME,
                                                                config_str))

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
    CP_SAVE_TAG = SEGNET_NAME

    TRAIN_BATCH_SIZE = 15

    DEPTH = 6
    NUM_CONV_LAYERS = [2, 2, 3, 3, 3, 3]
    NUM_FILTERS = [64, 128, 256, 256, 512, 512]

    SINGLE_BN = False
    CONV_ACT_BN = False
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

        return subparser

    @classmethod
    def __get_cmd_line_vars__(cls):
        cmd_line_vars = super().__get_cmd_line_vars__()
        cmd_line_vars.extend(['depth', 'num_conv_layers', 'num_filters'])
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
    CP_SAVE_TAG = UNET_NAME

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
        assert len(num_filters) == depth, "Number of filters must be specified for each depth"

        config_dict['NUM_FILTERS'] = num_filters

        return config_dict

class ResidualUNet(Config):
    """
    Configuration for 2D Residual U-Net architecture
    """
    CP_SAVE_TAG = 'res_unet'

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
    CP_SAVE_TAG = ENSEMBLE_UDS_NAME
    N_EPOCHS = 100

    def __init__(self, state='training', create_dirs=True):
        raise DeprecationWarning('This config is deprecated')
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class UNetMultiContrastConfig(UNetConfig):
    IMG_SIZE = (288, 288, 3)

    CP_SAVE_TAG = 'unet_2d_multi_contrast'

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

    CP_SAVE_TAG = 'unet_2_5d'
    N_EPOCHS = 20
    AUGMENT_DATA = True
    INITIAL_LEARNING_RATE = 1e-2

    DROP_RATE = 1.0
    DROP_FACTOR = 0.8

    # Train path - volumetric augmentation
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


class UNet3DConfig(UNetConfig):

    IMG_SIZE = (288, 288, 4, 1)

    CP_SAVE_TAG = 'unet_3d'
    N_EPOCHS = 20
    INITIAL_LEARNING_RATE = 1e-2

    DROP_RATE = 1.0
    DROP_FACTOR = 0.8

    TAG = 'oai_3d'

    SLICE_SUBSET = None  # 1 indexed inclusive - i.e. (5, 64) means slices [5, 64]

    NUM_FILTERS = [32, 64, 128, 256, 512, 1024]

    # Train path - volumetric augmentation
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

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

    CP_SAVE_TAG = 'deeplabv3_2_5d'
    N_EPOCHS = 100

    # Train path - volumetric augmentation
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


class AnisotropicUNetConfig(Config):
    """
    Configuration for 2D Anisotropic U-Net architecture
    """
    CP_SAVE_TAG = 'anisotropic_unet'

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
    CP_SAVE_TAG = 'refinenet'

    INITIAL_LEARNING_RATE = 1e-3

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


SUPPORTED_CONFIGS = [UNetConfig, SegnetConfig, DeeplabV3Config, ResidualUNet, AnisotropicUNetConfig, RefineNetConfig,
                     UNet3DConfig]


def get_config(config_cp_save_tag: str, create_dirs: bool=True):
    """
    Get config using config cp_save_tag
    :param config_cp_save_tag: config cp_save_tag
    :param create_dirs: if directory should be created
    :return: A Config instance
    """

    configs = SUPPORTED_CONFIGS
    for config in configs:
        if config.CP_SAVE_TAG == config_cp_save_tag:
            c = config(create_dirs=create_dirs)
            return c

    raise ValueError('config %s not found' % config_cp_save_tag)


def get_cp_save_tag(filepath: str):
    """
    Get cp_save_tag from a INI file
    :param filepath: filepath to INI file where config is stored
    :return: cp_save_tag specified in ini_filepath
    """
    config = configparser.ConfigParser()
    config.read(filepath)
    vars_dict = config['DEFAULT']
    return vars_dict['CP_SAVE_TAG']


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

