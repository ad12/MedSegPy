import config as MCONFIG
from config import DeeplabV3Config
from models import get_model

if __name__ == '__main__':
    MCONFIG.SAVE_PATH_PREFIX = './sample_data'
    # dil_rates_list = [(6, 12, 18), (2, 4, 6), (1, 9, 18)]
    #
    # for dil_rates in dil_rates_list:
    #     config = DeeplabV3Config()
    #     config.DIL_RATES = dil_rates
    #     get_model(config)
    #     K.clear_session()
    config = DeeplabV3Config()
    m = get_model(config)
    m.summary()
