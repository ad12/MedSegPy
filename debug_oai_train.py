from oai_train import train
import config as MCONFIG
from config import SegnetConfig
import argparse
import os

if __name__ == '__main__':
    #MCONFIG.SAVE_PATH_PREFIX = './sample_data'
    parser = argparse.ArgumentParser(description='Train OAI dataset')
    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use')
    args = parser.parse_args()
    gpu = args.gpu

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
    val_dict = {
                'N_EPOCHS': 1,
                'DEBUG': True,
                }

    config = SegnetConfig()

    train(config, val_dict)
