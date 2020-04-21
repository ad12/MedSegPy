"""Default engine activities.

Adapted from Facebook's detectron2.
https://github.com/facebookresearch/detectron2
"""
import argparse
import os

from fvcore.common.file_io import PathManager
import keras.backend as K

from medsegpy import glob_constants
from medsegpy.utils import dl_utils
from medsegpy.utils.collect_env import collect_env_info
from medsegpy.utils.logger import setup_logger


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="MedSegPy Training")
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--eval-only", action="store_true",
                        help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="number of gpus *per machine*")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the medsegpy logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    # Setup cuda visible devices.
    num_gpus = args.num_gpus
    if num_gpus > 0:
        gpu_ids = dl_utils.get_available_gpus(num_gpus)
        gpu_ids_tf_str = ",".join([str(g_id) for g_id in gpu_ids])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_tf_str
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    output_dir = cfg.OUTPUT_DIR
    PathManager.mkdirs(output_dir)

    setup_logger(output_dir, name="fvcore")
    logger = setup_logger(output_dir)

    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n")

    cfg.summary()
    cfg.save_config()

    if cfg.SEED is not None:
        logger.info("Using seed {}".format(cfg.SEED))
        glob_constants.SEED = cfg.SEED

    # Set image format to be (N, dim1, dim2, dim3, channel).
    K.set_image_data_format('channels_last')