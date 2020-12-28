"""Default engine activities.

Adapted from Facebook's detectron2.
https://github.com/facebookresearch/detectron2
"""
import argparse
import os
import shutil
import warnings

import keras.backend as K
import tensorflow as tf
from fvcore.common.file_io import PathManager

from medsegpy import glob_constants
from medsegpy.utils import dl_utils, env
from medsegpy.utils.collect_env import collect_env_info
from medsegpy.utils.logger import setup_logger
from medsegpy.utils.io_utils import format_exp_version


def config_exists(experiment_dir: str):
    return (
        os.path.isfile(os.path.join(experiment_dir, "config.ini"))
        or os.path.isfile(os.path.join(experiment_dir, "config.yaml"))
        or os.path.isfile(os.path.join(experiment_dir, "config.yml"))
    )


def default_argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="MedSegPy Training")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus"
    )
    # parser.add_argument(
    #     "--overwrite", action="store_true", help="overwrite previous experiment"
    # )
    parser.add_argument(
        "--debug", action="store_true", help="run in debug mode"
    )

    # Add option to execute non-eagerly in tensorflow 2
    if env.is_tf2():
        parser.add_argument(
            "--non-eagerly", action="store_true", help="run tensorflow non-eagerly"
        )

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
    4. Version experiments

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    # Make new experiment version if not evaluating
    # TODO: Add support to evaluate latest version (if found) when eval_only specified.
    cfg.OUTPUT_DIR = PathManager.get_local_path(cfg.OUTPUT_DIR)
    make_new_version = not (hasattr(args, "eval_only") and args.eval_only)
    if not make_new_version and not config_exists(cfg.OUTPUT_DIR):
        raise ValueError(
            f"Tried to evaluate on empty experiment directory. "
            f"{cfg.OUTPUT_DIR} does not exist."
        )

    if args.debug:
        os.environ["MEDSEGPY_RUN_MODE"] = "debug"
        cfg.OUTPUT_DIR = (
            os.path.join(cfg.OUTPUT_DIR, "debug")
            if os.path.basename(cfg.OUTPUT_DIR).lower() != "debug"
            else cfg.OUTPUT_DIR
        )
    else:
        cfg.OUTPUT_DIR = format_exp_version(cfg.OUTPUT_DIR, new_version=make_new_version)

    # Setup cuda visible devices.
    num_gpus = args.num_gpus
    if num_gpus > 0:
        gpu_ids = dl_utils.get_available_gpus(num_gpus)
        gpu_ids_tf_str = ",".join([str(g_id) for g_id in gpu_ids])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_tf_str
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Set seed.
    if cfg.SEED == -1:
        cfg.SEED = env.generate_seed()

    # Set experiment name.
    cfg.EXP_NAME = default_exp_name(cfg)

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
    K.set_image_data_format("channels_last")

    # Non-eager execution in tf2
    if env.is_tf2() and args.non_eagerly:
        if env.tf_version() >= (2, 3):
            tf.config.run_functions_eagerly(False)
            logger.info("Disabling eager execution...")
        else:
            logger.warning(
                "Eager mode has not been disabled. May have to disable manually in the model"
            )


def default_exp_name(cfg):
    """Extracts default experiment name from the config.

    `cfg.EXP_NAME` if exists. If basename starts with "version" or "debug", take both parent directory name
    and version name to make experiment name (e.g. "my_exp/version_001").

    Returns:
        exp_name (str): The default convention for naming experiments.
    """
    exp_name = cfg.get("EXP_NAME")
    if not exp_name:
        warnings.warn("EXP_NAME not specified. Defaulting to basename...")
        basename = os.path.basename(cfg.OUTPUT_DIR)
        if basename.startswith("version") or basename.startswith("debug"):
            exp_name = f"{os.path.basename(os.path.dirname(cfg.OUTPUT_DIR))}/{basename}"
        else:
            exp_name = basename
    return exp_name
