import argparse
import ast
import logging
import os

os.environ["MSK_SEG_NETWORKS_PROJECT"] = "tech-considerations_v3"

from medsegpy.oai_test import test_dir, get_valid_subdirs
from medsegpy.utils import dl_utils
from medsegpy.utils.metric_utils import SegMetric

logger = logging.getLogger(__name__)


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')
    parser.add_argument('--voxel_spacing', type=str, nargs='?', default=None,
                        help='voxel spacing. eg. \'(0.5, 0.5, 2)\'')
    parser.add_argument("--metrics", nargs="*", default=None,
                        choices=SegMetric.__members__.keys(),
                        help="metrics to use for evaluation",
                        )
    parser.add_argument("--num_gpus",
                        default=1,
                        type=int,
                        help="number of gpus to use. defaults to 1")
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
    if "tag" in vargin and vargin['tag']:
        config_dict['TAG'] = vargin['tag']
        config_dict['TEST_RESULTS_FOLDER_NAME'] = 'test_results_%s' % vargin['tag']

    return config_dict


if __name__ == '__main__':
    logger = logging.getLogger("medsegpy.nn_test.{}".format(__name__))
    base_parser = argparse.ArgumentParser(description='Run inference (testing)')
    add_testing_arguments(base_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    config_filepath = vargin["dirpath"][0]
    if not os.path.isdir(config_filepath):
        raise NotADirectoryError(
            "Directory {} does not exist".format(config_filepath)
        )

    num_gpus = args.num_gpus
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if num_gpus > 0:
        gpu_ids = dl_utils.get_available_gpus(num_gpus)
        gpu_ids_tf_str = ",".join([str(g_id) for g_id in gpu_ids])
        logger.info("Using {} GPU(s): {}".format(num_gpus, gpu_ids_tf_str))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_tf_str
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

    metrics = args.metrics
    metrics = [SegMetric[m] for m in metrics] if metrics else None

    for dp in test_dirpaths:
        test_dir(
            dp,
            vals_dict=create_config_dict(vargin),
            save_h5_data=vargin['save_h5_data'],
            voxel_spacing=voxel_spacing,
            metrics=metrics,
        )
