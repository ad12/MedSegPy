import argparse
import logging
import os

import keras.backend as K

from medsegpy import config as _config
from medsegpy.data.im_gens import get_generator
from medsegpy.evaluation import build_evaluator
from medsegpy.evaluation import inference_on_dataset
from medsegpy.modeling import get_model
from medsegpy.utils import dl_utils
from medsegpy.utils.logger import setup_logger
from medsegpy.utils.metric_utils import SegMetric

logger = logging.getLogger(__name__)


def add_testing_arguments(parser: argparse.ArgumentParser):
    parser.add_argument('--dirpath', metavar='dp', type=str, nargs=1,
                        help='path to config to test')
    parser.add_argument("--metrics", nargs="*", default=None,
                        choices=SegMetric.__members__.keys(),
                        help="metrics to use for evaluation",
                        )
    parser.add_argument("--num_gpus",
                        default=1,
                        type=int,
                        help="number of gpus to use. defaults to 1")
    parser.add_argument('--batch_size', default=72, type=int, nargs='?')
    parser.add_argument('--save_h5_data', action='store_true',
                        help='save raw data (predictions, binary labels, masks)')
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
    config_dict["TEST_METRICS"] = vargin["metrics"]

    return config_dict


def test_dir(
    experiment_dir,
    config=None,
    vals_dict=None,
    weights_path=None,
    save_h5_data=False,
):
    """Run inference on experiment located in `dirpath`.

    Args:
        experiment_dir (str): path to directory storing experiment training
            outputs.
        config: a Config object
        vals_dict: a dictionary of config parameters to change (default = None).
            e.g. `{'INITIAL_LEARNING_RATE': 1e-6, 'USE_STEP_DECAY': True}`
        weights_path: path to best weights (default = None). If None,
            automatically search dirpath for the best weight path.
        save_h5_data (:obj:`bool`, optional): If `True`, save predictions (0-1),
            labels (binarized), and ground truth (binarized) in h5 file.
    """
    # Create config, if not provided.
    config_filepath = os.path.join(experiment_dir, 'config.ini')
    if not config:
        config = _config.get_config(
            _config.get_cp_save_tag(config_filepath),
            create_dirs=False,
        )

    # Get best weight path
    if weights_path is None:
        weights_path = dl_utils.get_weights(experiment_dir)

    config.load_config(config_filepath)
    config.TEST_WEIGHT_PATH = weights_path

    # Initialize logger.
    setup_logger(config.OUTPUT_DIR)
    logger.info("OUTPUT_DIR: {}".format(config.OUTPUT_DIR))
    logger.info("Config: {}".format(config_filepath))
    logger.info("Best weights: {}".format(weights_path))

    if vals_dict is not None:
        for key in vals_dict.keys():
            val = vals_dict[key]
            config.set_attr(key, val)

    config.change_to_test()

    test_gen = get_generator(config)

    K.set_image_data_format('channels_last')
    model = get_model(config)
    model.load_weights(config.TEST_WEIGHT_PATH)

    evaluator = build_evaluator(
        config.TEST_DATASET,
        config,
        output_dir=config.TEST_RESULT_PATH,
        save_raw_data=save_h5_data
    )

    inference_on_dataset(model, test_gen, evaluator)

    K.clear_session()


if __name__ == '__main__':
    logger = logging.getLogger("medsegpy.nn_test.{}".format(__name__))
    base_parser = argparse.ArgumentParser(description='Run inference')
    add_testing_arguments(base_parser)

    # Parse input arguments
    args = base_parser.parse_args()
    vargin = vars(args)

    root_dir = vargin["dirpath"][0]
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(
            "Directory {} does not exist".format(root_dir)
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

    test_dirpaths = [root_dir]
    if recursive:
        test_dirpaths = dl_utils.get_valid_subdirs(root_dir, not overwrite)
    else:
        test_dirpaths = [root_dir]

    for dp in test_dirpaths:
        test_dir(
            dp,
            vals_dict=create_config_dict(vargin),
            save_h5_data=vargin['save_raw_data'],
        )
