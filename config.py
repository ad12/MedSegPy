import os
from time import gmtime, strftime

import mri_utils
import utils


class Config():
    VERSION = 1

    DEBUG = False

    # Model architecture path
    PLOT_MODEL_PATH = utils.check_dir('./model_imgs')

    # Training and validation image size
    IMG_SIZE = (288, 288, 1)

    # Training epochs
    N_EPOCHS = 30

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
    TRAIN_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/train_aug_2d'
    VALID_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/valid_2d'
    TEST_PATH = '/bmrNAS/people/akshay/dl/oai_data/oai_aug/test_2d'

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
        self.CP_SAVE_PATH = utils.check_dir(os.path.join('/bmrNAS/people/arjun/msk_seg_networks/oai_data', self.CP_SAVE_TAG, prefix))
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

        # Save optimizer state
        filename = os.path.join(self.CP_SAVE_PATH, 'config.dat')
        m_config = self.to_dict_w_opt(model)
        utils.save_pik(m_config, filename)

        # Save as object to make it easy to load
        self.M_CONFIG = m_config
        filename = os.path.join(self.CP_SAVE_PATH, 'config_obj.dat')
        utils.save_pik(self, filename)

    def load_config(self, ini_filepath):
        vars_dict = utils.load_config(ini_filepath)

        if(vars_dict['CP_SAVE_PATH'] != self.CP_SAVE_PATH):
            raise ValueError('Wrong config. Expected %s' % str(vars_dict['CP_SAVE_PATH']))

        for key in vars_dict.keys():
            self.__setattr__(key, vars_dict[key])

    def to_dict_w_opt(self, model):
        """Serialize a model and add the config of the optimizer
        """
        if model is None:
            return None
        config = dict()
        config_m = model.get_config()
        config['config'] = {'class_name': model.__class__.__name__,'config': config_m,}
        if hasattr(model, 'optimizer'):
            config['optimizer'] = model.optimizer.get_config()

        return config

    def model_from_dict_w_opt(self, model_dict):
        """ Return model and optimizer in previous state
        """
        from keras import optimizers
        optimizer_params = dict([(k,v) for k,v in model_dict.get('optimizer').items()])
        optimizer = optimizers.get(optimizer_params)

        return optimizer

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
    CP_SAVE_TAG = 'deeplabv3_2d'
    DIL_RATES = (1, 1, 1)
    AT_DIVISOR = 2

    FINE_TUNE = False
    INIT_WEIGHT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-08-21-07-03-24/deeplabv3_2d_weights.018-0.1191.h5'

    # Test weight path is divisor 2
    TEST_WEIGHT_PATH = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/deeplabv3_2d/2018-08-21-07-03-24/deeplabv3_2d_weights.018-0.1191.h5'

    OS = 16
    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class SegnetConfig(Config):
    CP_SAVE_TAG = 'segnet_2d'

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
    CP_SAVE_TAG='unet_2d'
    TEST_WEIGHT_PATH = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'
    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)


class EnsembleUDSConfig(Config):
    CP_SAVE_TAG = 'ensemble_uds'
    DEEPLAB_INIT_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/deeplab_2d_end-to-end/2018-08-15-06-46-12/deeplab_2d_end-to-end_weights.019-0.1253.h5'
    UNET_INIT_WEIGHTS = '/bmrNAS/people/akshay/dl/oai_data/unet_2d/select_weights/unet_2d_fc_weights.004--0.8968.h5'
    SEGNET_INIT_WEIGHTS = '/bmrNAS/people/arjun/msk_seg_networks/oai_data/segnet_2d/2018-08-18-19-55-54/segnet_2d_weights.005-0.3353.h5'
    N_EPOCHS = 40

    def __init__(self, state='training', create_dirs=True):
        super().__init__(self.CP_SAVE_TAG, state, create_dirs=create_dirs)



