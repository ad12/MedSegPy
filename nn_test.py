from __future__ import print_function, division

import argparse
import os

from oai_test import test_dir
import nn_interp_test

def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use. default=0')
    parser.add_argument('--cpu', action='store_const', default=False, const=True,
                        help='use cpu. will overridie `-g` gpu flag')

    parser.add_argument('--batch_size', default=72, type=int, nargs='?')
    parser.add_argument('--save_h5_data', action='store_const', const=True, default=False,
                        help='save h5 data')
    parser.add_argument('--tag', default=None, nargs='?', type=str,
                        help='change tag for inference')
    parser.add_argument('--img_size', default=None, nargs='?')


def create_config_dict(vargin):
    config_dict = {'TEST_BATCH_SIZE': vargin['batch_size']}
    if vargin['tag']:
        config_dict['TAG'] = vargin['tag']
        config_dict['TEST_RESULTS_FOLDER_NAME'] = 'test_results_%s' % vargin['tag']

    return config_dict


if __name__ == '__main__':
    base_parser = argparse.ArgumentParser(description='Run inference (testing)')
    add_testing_arguments(base_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    config_filepath = vargin['dirpath'][0]
    if not os.path.isdir(config_filepath):
        raise NotADirectoryError('Directory %s does not exist.' % config_filepath)

    gpu = args.gpu
    cpu = args.cpu

    if not cpu:
        print('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    else:

        os.environ["CUDA_VISIBLE_DEVICES"] = ""


    test_dir(config_filepath, vals_dict=create_config_dict(vargin), save_h5_data=vargin['save_h5_data'])
