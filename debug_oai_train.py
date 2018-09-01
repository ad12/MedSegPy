from oai_train import train
import config as MCONFIG
from config import SegnetConfig


if __name__ == '__main__':
    #MCONFIG.SAVE_PATH_PREFIX = './sample_data'

    val_dict = {
                'N_EPOCHS': 1,
                'DEBUG': True,
                }

    config = SegnetConfig()

    train(config, val_dict)
