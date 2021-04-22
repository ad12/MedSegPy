import importlib
import importlib.util
import logging
import os
import random
import sys
from datetime import datetime
from typing import Tuple

import numpy as np
import tensorflow as tf

__all__ = []

_ENV_SETUP_DONE = False
_SETTINGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.settings"))
_TF_VERSION = None


def generate_seed():
    """Generate a random seed."""
    seed = os.getpid() + int(datetime.now().strftime("%S%f")) + int.from_bytes(os.urandom(2), "big")
    logger = logging.getLogger(__name__)
    logger.info("Generated random seed {}".format(seed))
    return seed


def seed_all_rng(seed=None):
    """Set random seed for RNG in tensorflow, numpy, and python.

    Args:
        seed (int, optional): The random seed. If not specified, this function
            generates a random seed using :func:`generate_seed`.
    """
    logger = logging.getLogger(__name__)
    if seed is None:
        seed = generate_seed()

    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except AttributeError:
        tf.random.set_random_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Using random seed {}".format(seed))


def is_debug():
    return os.environ.get("MEDSEGPY_RUN_MODE", "") == "debug"


# from https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path  # noqa
def _import_file(module_name, file_path, make_importable=False):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[module_name] = module
    return module


def _configure_libraries():
    """
    Configurations for some libraries.
    """
    # An environment option to disable `import cv2` globally,
    # in case it leads to negative performance impact
    disable_cv2 = int(os.environ.get("MEDSEGPY_DISABLE_CV2", False))
    if disable_cv2:
        sys.modules["cv2"] = None
    else:
        # Disable opencl in opencv since its interaction with cuda often
        # has negative effects
        # This envvar is supported after OpenCV 3.4.0
        os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"
        try:
            import cv2

            if int(cv2.__version__.split(".")[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ImportError:
            pass


def setup_environment():
    """Perform environment setup work. The default setup is a no-op, but this
    function allows the user to specify a Python source file or a module in
    the $MEDSEGPY_ENV_MODULE environment variable, that performs
    custom setup work that may be necessary to their computing environment.
    """
    global _ENV_SETUP_DONE
    if _ENV_SETUP_DONE:
        return
    _ENV_SETUP_DONE = True

    _configure_libraries()

    custom_module_path = os.environ.get("MEDSEGPY_ENV_MODULE")

    if custom_module_path:
        setup_custom_environment(custom_module_path)
    else:
        # The default setup is a no-op
        pass


def setup_custom_environment(custom_module):
    """
    Load custom environment setup by importing a Python source file or a
    module, and run the setup function.
    """
    if custom_module.endswith(".py"):
        module = _import_file("medsegpy.utils.env.custom_module", custom_module)
    else:
        module = importlib.import_module(custom_module)
    assert hasattr(module, "setup_environment") and callable(module.setup_environment), (
        "Custom environment module defined in {} does not have the "
        "required callable attribute 'setup_environment'."
    ).format(custom_module)
    module.setup_environment()


def supports_wandb():
    return "wandb" in sys.modules and not is_debug()


def tf_version() -> Tuple[int, ...]:
    global _TF_VERSION
    if not _TF_VERSION:
        import tensorflow as tf

        _TF_VERSION = [int(x) for x in tf.__version__.split(".")[:2]]
    return tuple(_TF_VERSION)


def is_tf2():
    """Returns `True` if running tensorflow 2.X"""
    version = tf_version()
    return version[0] == 2


def settings_dir():
    return os.environ.get("MEDSEGPY_SETTINGS", _SETTINGS_DIR)
