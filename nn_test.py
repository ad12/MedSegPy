from __future__ import print_function, division

import argparse
import os

from oai_test import test_dir


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')

    parser.add_argument('-g', '--gpu', metavar='G', type=str, nargs='?', default='0',
                        help='gpu id to use. default=0')

    parser.add_argument('--batch_size', default=72, type=int, nargs='?')
    parser.add_argument('--save_h5_data', action='store_const', const=True, default=False,
                        help='save h5 data')

    # parser.add_argument('--save_h5_data', action='store_const', default=False, const=True,
    #                     help='save ground truth and prediction data in h5 format')


def create_config_dict(vargin):
    return {'TEST_BATCH_SIZE': vargin['batch_size']}


if __name__ == '__main__':
    base_parser = argparse.ArgumentParser(description='Run inference (testing)')
    add_testing_arguments(base_parser)
    # arg_subparser = base_parser.add_subparsers(help='supported configs for different architectures', dest='config')
    # subparsers = MCONFIG.init_cmd_line_parser(arg_subparser)
    #
    # for s_parser in subparsers:
    #     add_testing_arguments(s_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    config_filepath = vargin['dirpath'][0]
    if not os.path.isdir(config_filepath):
        raise NotADirectoryError('Directory %s does not exist.' % config_filepath)

    gpu = args.gpu

    print('Using GPU %s' % gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    test_dir(config_filepath, vals_dict=create_config_dict(vargin), save_h5_data=vargin['save_h5_data'])
