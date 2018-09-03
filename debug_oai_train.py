from oai_train import train
import config as MCONFIG
from config import SegnetConfig, DeeplabV3Config
from models import get_model
import argparse
import os
from keras import backend as K

if __name__ == '__main__':
    MCONFIG.SAVE_PATH_PREFIX = './sample_data'
    # dil_rates_list = [(6, 12, 18), (2, 4, 6), (1, 9, 18)]
    #
    # for dil_rates in dil_rates_list:
    #     config = DeeplabV3Config()
    #     config.DIL_RATES = dil_rates
    #     get_model(config)
    #     K.clear_session()
    config = SegnetConfig()
    config.DEPTH = 7
    config.NUM_CONV_LAYERS = [2]*7
    config.NUM_FILTERS = [16, 32, 64, 128, 256, 512, 1024]
    m = get_model(config)
    m.summary()