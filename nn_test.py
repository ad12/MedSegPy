import argparse
import ast
import logging
import os

os.environ["MSK_SEG_NETWORKS_PROJECT"] = "tech-considerations_v3"

from oai_test import test_dir, get_valid_subdirs

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')
    parser.add_argument('--voxel_spacing', type=str, nargs='?', default=None,
                        help='voxel spacing. eg. \'(0.5, 0.5, 2)\'')
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

    parser.add_argument('-r', '--recursive', action='store_true',
                        help='recursively analyze all directories')
    parser.add_argument('-f', '--force', action='store_true',
                        help='overwrite existing test folders')


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
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if not cpu:
        logger.info('Using GPU %s' % gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    recursive = args.recursive
    overwrite = args.force
    voxel_spacing = args.voxel_spacing

    test_dirpaths = [config_filepath]
    if recursive:
        test_dirpaths = get_valid_subdirs(config_filepath, not overwrite)

    if voxel_spacing:
        voxel_spacing = ast.literal_eval(voxel_spacing)
    
    for dp in test_dirpaths:
        test_dir(dp, vals_dict=create_config_dict(vargin), save_h5_data=vargin['save_h5_data'], voxel_spacing=voxel_spacing)
