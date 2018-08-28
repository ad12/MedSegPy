import os
from time import gmtime, strftime

import mri_utils
import utils


DEEPLABV3_NAME = 'deeplabv3_2d'
SEGNET_NAME = 'segnet_2d'
UNET_NAME = 'unet_2d'
ENSEMBLE_UDS_NAME = 'ensemble_uds'

# This is the default save path prefix - please change if you desire something else
SAVE_PATH_PREFIX = '/bmrNAS/people/arjun/msk_seg_networks/oai_data'

class Config():
    VERSION = 2

    PIDS = None

    DEBUG = False

    # Model architecture path
    PLOT_MODEL_PATH = utils.check_dir('./model_imgs')

    # Training and validation image size
    IMG_SIZE = (288, 288, 1)

    # Training epochs
    N_EPOCHS = 20

    TRAIN_BATCH_SIZE = 12
    VALID_BATCH_SIZE = 35
    TEST_BATCH_SIZE = 72

    INITIAL_LEARNING_RATE = 2e-2
    MIN_LEARNING_RATE = 1e-8
    DROP_FACTOR = 0.7
    DROP_RATE = 1.0

    # Tissues to render
    TISSUES = [mri_utils.MASK_FEMORAL_CARTILAGE]
    NUM_CLASSES = len(TISSUES)

    # File Types
    FILE_TYPES = ['im']

    # Transfer Learning
    FINE_TUNE = False
    INIT_WEIGHT_PATH = ''

    # Dataset Paths
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/train_aug/'
    VALID_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/valid/'
    TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/test'

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
    TAG = 'oai_aug'

    # Restrict number of files learned. Default is all []
    LEARN_FILES = []

    def __init__(self, cp_save_tag, state='training', create_dirs=True):

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
                self.TEST_RESULT_PATH = utils.check_dir(os.path.join('./test_results', self.CP_SAVE_TAG, self.TAG, self.DATE_TIME_STR))

    def init_training_paths(self, prefix):
        self.CP_SAVE_PATH = utils.check_dir(os.path.join(SAVE_PATH_PREFIX, self.CP_SAVE_TAG, prefix))
        self.PIK_SAVE_PATH = os.path.join(self.CP_SAVE_PATH, 'pik_data.dat')
        self.PIK_SAVE_PATH_DIR = utils.check_dir(os.path.dirname(self.PIK_SAVE_PATH))
        self.TF_LOG_DIR = utils.check_dir(os.path.join(self.CP_SAVE_PATH, 'tf_log'))

    def save_config(self, model=None):
        """
        Save params of config to ini file
        :param model:
        :return:
        """

        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        filename = os.path.join(self.CP_SAVE_PATH, 'config.ini')

        config_vars = dict()
        for m_var in members:
            config_vars[m_var] = getattr(self, m_var)
        utils.save_config(config_vars, filename)

        # Save as object to make it easy to load
        filename = os.path.join(self.CP_SAVE_PATH, 'config_obj.dat')
        utils.save_pik(self, filename)

    def load_config(self, ini_filepath):
        vars_dict = utils.load_config(ini_filepath)

        if(vars_dict['CP_SAVE_TAG'] != self.CP_SAVE_TAG):
            raise ValueError('Wrong config. Expected %s' % str(vars_dict['CP_SAVE_TAG']))

        for key in vars_dict.keys():
            upper_case_key = str(key).upper()
            if not hasattr(self, upper_case_key):
                raise ValueError('Key %s does not exist. Please make sure all variable names are fully capitalized' % upper_case_key)

            # all data is of type string, but we need to cast back to original data type
            data_type = type(getattr(self, upper_case_key))
            var_converted = utils.convert_data_type(vars_dict[key], data_type)
            self.__setattr__(str(key).upper(), var_converted)

    def set_attr(self, attr, val):
        if type(attr) is not str:
            raise ValueError('attr must be of type str')

        if not hasattr(self, attr):
            raise ValueError('The attribute %s does not exist' % attr)
        curr_val = self.__getattribute__(attr)

        if (type(val) != type(curr_val)):
            raise ValueError('Input value is of type %s. Expected %s' %(str(type(val)), str(type(curr_val))))

        self.__setattr__(attr, val)

    def init_fine_tune(self, init_weight_path):
        if (self.state != 'training'):
            raise ValueError('Must be in training state')

        self.FINE_TUNE = True
        self.INIT_WEIGHT_PATH = init_weight_path

        prefix = os.path.join(self.DATE_TIME_STR, 'fine_tune')
        self.init_training_paths(prefix)

    def change_to_test(self):
        self.state = 'testing'
        self.TEST_RESULT_PATH = utils.check_dir(os.path.join(self.CP_SAVE_PATH, 'test_results'))


class DeeplabV3Config(Config):
    CP_SAVE_TAG = DEEPLABV3_NAME
    DIL_RATES = (1, 1, 1)
    AT_DIVISOR = 2

    FINE_TUNE = False
    INIT_WEIGHT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-08-21-07-03-24/deeplabv3_2d_weights.018-0.1191.h5'

    # Test weight path is divisor 2
    TEST_WEIGHT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-08-21-07-03-24/deeplabv3_2d_weights.018-0.1191.h5'

    OS = 16
    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)

    def change_to_test(self):
        self.state = 'testing'
        config_tuple = (self.OS, ) + self.DIL_RATES
        config_str = '%d_%d-%d-%d' % config_tuple
        self.TEST_RESULT_PATH = utils.check_dir(os.path.join(self.CP_SAVE_PATH, 'test_results', config_str))


class SegnetConfig(Config):
    CP_SAVE_TAG = SEGNET_NAME

    TRAIN_BATCH_SIZE = 15
    #INITIAL_LEARNING_RATE = 2e-6
    FINE_TUNE=False
    INIT_WEIGHT_PATH='/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-08-18-19-55-54/segnet_2d_weights.005-0.3353.h5'
    TEST_WEIGHT_PATH='/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-08-18-19-55-54/segnet_2d_weights.005-0.3353.h5'

    DEPTH = 6
    NUM_CONV_LAYERS = [2, 2, 3, 3, 3, 3]
    NUM_FILTERS = [64,128,256,256,512,512]

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class UNetConfig(Config):
    CP_SAVE_TAG = UNET_NAME
    TEST_WEIGHT_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'
    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class EnsembleUDSConfig(Config):
    CP_SAVE_TAG = ENSEMBLE_UDS_NAME
    DEEPLAB_INIT_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/deeplab_2d_end-to-end/2018-08-15-06-46-12/deeplab_2d_end-to-end_weights.019-0.1253.h5'
    UNET_INIT_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'
    SEGNET_INIT_WEIGHTS = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-08-18-19-55-54/segnet_2d_weights.005-0.3353.h5'
    N_EPOCHS = 40

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class UNetMultiContrastConfig(Config):
    IMG_SIZE = (288, 288, 3)

    CP_SAVE_TAG = 'unet_2d_multi_contrast'

    # Whether to load weights from original unet model
    INIT_UNET_2D = True
    INIT_UNET_2D_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


