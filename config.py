import configparser
import os
import warnings
from time import gmtime, strftime

import glob_constants as glc
import mri_utils
from losses import DICE_LOSS, CMD_LINE_SUPPORTED_LOSSES, get_training_loss_from_str
from utils import io_utils
import utils.utils as utils
from losses import DICE_LOSS, CMD_LINE_SUPPORTED_LOSSES, get_training_loss_from_str

DEPRECATED_KEYS = ['NUM_CLASSES']

DEEPLABV3_NAME = 'deeplabv3_2d'
SEGNET_NAME = 'segnet_2d'
UNET_NAME = 'unet_2d'
ENSEMBLE_UDS_NAME = 'ensemble_uds'

# This is the default save path prefix - please change if you desire something else
SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/oai_data'

CMD_LINE_VARS = ['n_epochs', 'augment_data',
                 'use_step_decay', 'initial_learning_rate', 'min_learning_rate', 'drop_factor', 'drop_rate',
                 'use_early_stopping', 'early_stopping_min_delta', 'early_stopping_patience',
                 'early_stopping_criterion',
                 'train_batch_size', 'valid_batch_size', 'test_batch_size',
                 'loss', 'include_background',
                 'img_size']

SUPPORTED_CONFIGS = [DEEPLABV3_NAME, SEGNET_NAME, UNET_NAME]


def init_cmd_line_parser(parser):
    # Number of epochs
    parser.add_argument('--n_epochs', metavar='E', type=int, default=None, nargs='?',
                        help='Number of training epochs')

    # Augment data
    parser.add_argument('--augment_data', type=bool, default=False, action='store_const', const=True,
                        help='Use augmented data for training')

    # Learning rate step decay
    parser.add_argument('--use_step_decay', type=bool, default=False, action='store_const', const=True,
                        help='use learning rate step decay')
    parser.add_argument('--initial_learning_rate', metavar='LR', type=float, default=1e-4, nargs='?',
                        help='initial learning rate')
    parser.add_argument('--min_learning_rate', metavar='mLR', type=float, default=1e-8, nargs='?',
                        help='minimum learning rate during decay')
    parser.add_argument('--drop_factor', metavar='DF', type=float, default=0.7, nargs='?',
                        help='drop factor for learning rate decay')
    parser.add_argument('--drop_rate', metavar='DR', type=int, default=1.0, nargs='?',
                        help='drop rate for learning rate decay')

    # Early stopping
    parser.add_argument('--use_early_stopping', type=bool, default=False, action='store_const', const=True,
                        help='use learning rate step decay')
    parser.add_argument('--early_stopping_min_delta', metavar='D', type=float, default=0.0, nargs='?',
                        help='minimum change in the monitored quantity to qualify as an improvement, '
                             'i.e. an absolute change of less than min_delta, will count as no improvement.')
    parser.add_argument('--early_stopping_patience', metavar='P', type=int, default=0, nargs='?',
                        help='number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_criterion', metavar='C', type=str, default='val_loss', nargs='?',
                        help='criterion to monitor for early stopping')

    # Batch size
    parser.add_argument('--train_batch_size', metavar='trBS', type=int, default=12, nargs='?',
                        help='training batch size')
    parser.add_argument('--valid_batch_size', metavar='vBS', type=int, default=35, nargs='?',
                        help='drop rate for learning rate decay')
    parser.add_argument('--test_batch_size', metavar='tBS', type=int, default=72, nargs='?',
                        help='drop rate for learning rate decay')

    # Loss function
    parser.add_argument('--loss', metavar='L', type=str, default='DICE_LOSS', nargs='?',
                        choices=CMD_LINE_SUPPORTED_LOSSES,
                        help='loss function')

    # Include background
    parser.add_argument('--include_background', type=bool, default=False, action='store_const', const=True,
                        help='loss function')

    # Image size
    parser.add_argument('--img_size', type=tuple, default=(288, 288, 1), nargs='?',
                        help='loss function')


def parse_cmd_line(vargin):
    config_dict = dict()
    for skey in CMD_LINE_VARS:
        if skey not in vargin.keys():
            continue

        c_skey = skey.upper()
        val = vargin[skey]

        if skey == 'loss':
            val = get_training_loss_from_str(vargin[skey].upper())

        if skey == 'img_size':
            assert type(val) is tuple and len(val) == 3

        config_dict[c_skey] = val

    return config_dict


class Config():
    VERSION = 4

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
    TRAIN_FILES_CV = None
    VALID_FILES_CV = None
    TEST_FILES_CV = None

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

    def __init__(self, cp_save_tag, state='training', create_dirs=True):
        self.SEED = glc.SEED
        if state not in ['testing', 'training']:
            raise ValueError('state must either be \'training\' or \'testing\'')

        self.DATE_TIME_STR = strftime("%Y-%m-%d-%H-%M-%S", gmtime())

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

    def init_cross_validation(self, train_files, valid_files, test_files, cv_tag):
        assert self.STATE == 'training', "To initialize cross-validation, must be in training state"

        self.USE_CROSS_VALIDATION = True
        self.TRAIN_FILES_CV = train_files
        self.VALID_FILES_CV = valid_files
        self.TEST_FILES_CV = test_files

        assert self.CP_SAVE_PATH, "CP_SAVE_PATH must be defined - call init_training_paths prior to calling this function"

        self.CP_SAVE_PATH = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH, cv_tag))
        self.PIK_SAVE_PATH = os.path.join(self.CP_SAVE_PATH, 'pik_data.dat')
        self.PIK_SAVE_PATH_DIR = io_utils.check_dir(os.path.dirname(self.PIK_SAVE_PATH))
        self.TF_LOG_DIR = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH, 'tf_log'))

    def save_config(self):
        """
        Save params of config to ini file
        """

        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
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

        if (curr_val is not None and (type(val) != type(curr_val))):
            raise ValueError('Input value is of type %s. Expected %s' % (str(type(val)), str(type(curr_val))))

        self.__setattr__(attr, val)

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

    def change_to_test(self):
        """
        Initialize testing state
        """
        self.STATE = 'testing'
        self.TEST_RESULT_PATH = io_utils.check_dir(os.path.join(self.CP_SAVE_PATH, self.TEST_RESULTS_FOLDER_NAME))

    def summary(self, additional_vars=[]):
        """
        Print config summary
        :param additional_vars: additional list of variables to print
        :return:
        """

        summary_vals = ['CP_SAVE_TAG']

        if self.STATE == 'training':
            summary_vals.extend(
                ['N_EPOCHS', 'AUGMENT_DATA', 'LOSS', 'TRAIN_BATCH_SIZE', 'VALID_BATCH_SIZE', 'USE_STEP_DECAY',
                 'INITIAL_LEARNING_RATE', 'MIN_LEARNING_RATE', 'DROP_FACTOR', 'DROP_RATE', 'FINE_TUNE'])
            if self.FINE_TUNE:
                summary_vals.extend(['INIT_WEIGHT_PATH'])
        else:
            summary_vals.extend(['TEST_RESULT_PATH', 'TEST_WEIGHT_PATH', 'TEST_BATCH_SIZE'])

        summary_vals.extend(additional_vars)

        print('')
        print('==' * 40)
        print("Config Summary")
        print('==' * 40)

        for attr in summary_vals:
            print(attr + ": " + str(self.__getattribute__(attr)))

        print('==' * 40)
        print('')

    def get_num_classes(self):
        if (self.INCLUDE_BACKGROUND):
            return len(self.TISSUES) + 1

        return len(self.TISSUES)

    def num_neighboring_slices(self):
        return None


class DeeplabV3Config(Config):
    CP_SAVE_TAG = DEEPLABV3_NAME

    OS = 16
    DIL_RATES = (2, 4, 6)
    AT_DIVISOR = 2
    DROPOUT_RATE = 0.1

    FINE_TUNE = False
    INIT_WEIGHT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-08-21-07-03-24/deeplabv3_2d_weights.018-0.1191.h5'

    # Test weight path is divisor 2
    TEST_WEIGHT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-08-21-07-03-24/deeplabv3_2d_weights.018-0.1191.h5'

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    def change_to_test(self):
        self.state = 'testing'
        config_tuple = (self.OS,) + self.DIL_RATES
        config_str = '%d_%d-%d-%d' % config_tuple
        self.TEST_RESULT_PATH = utils.check_dir(
            os.path.join(self.CP_SAVE_PATH, self.TEST_RESULTS_FOLDER_NAME, config_str))

    def summary(self, additional_vars=[]):
        summary_attrs = ['OS', 'DIL_RATES', 'DROPOUT_RATE']
        super().summary(summary_attrs)


class SegnetConfig(Config):
    CP_SAVE_TAG = SEGNET_NAME

    TRAIN_BATCH_SIZE = 15
    FINE_TUNE = False
    INIT_WEIGHT_PATH = ''
    TEST_WEIGHT_PATH = ''

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


class UNetConfig(Config):
    CP_SAVE_TAG = UNET_NAME
    TEST_WEIGHT_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'

    INIT_UNET_2D = False

    USE_STEP_DECAY = True
    INITIAL_LEARNING_RATE = 2e-2
    DROP_FACTOR = 0.8 ** (1 / 5)
    DROP_RATE = 1.0
    TRAIN_BATCH_SIZE = 35

    DEPTH = 6
    NUM_FILTERS = None

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class EnsembleUDSConfig(Config):
    CP_SAVE_TAG = ENSEMBLE_UDS_NAME
    AUGMENT_DATA = False
    N_EPOCHS = 100

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class UNetMultiContrastConfig(UNetConfig):
    IMG_SIZE = (288, 288, 3)

    CP_SAVE_TAG = 'unet_2d_multi_contrast'

    # Whether to load weights from original unet model
    INIT_UNET_2D = True
    INIT_UNET_2D_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class UNet2_5DConfig(UNetConfig):
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


class DeeplabV3_2_5DConfig(DeeplabV3Config):
    IMG_SIZE = (288, 288, 3)

    CP_SAVE_TAG = 'deeplabv3_2_5d'
    N_EPOCHS = 100
    AUGMENT_DATA = False

    # Train path - volumetric augmentation
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/vol_aug/train_sag/'

    def num_neighboring_slices(self):
        return self.IMG_SIZE[2]


def save_config(a_dict, filepath):
    """
    Save information in a dictionary
    :param a_dict: a dictionary of information to save
    :param filepath: a string
    :return:
    """
    config = configparser.ConfigParser(a_dict)

    with open(filepath, 'w+') as configfile:
        config.write(configfile)


def load_config(filepath):
    """
    Read in information saved using save_config
    :param filepath: a string
    :return: a dictionary of Config params
    """
    config = configparser.ConfigParser()
    config.read(filepath)

    return config['DEFAULT']
