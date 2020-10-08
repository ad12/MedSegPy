import os
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from tabulate import tabulate


def _pretty_print_keras_config(env_str):
    # TODO (TF2.X)
    if not hasattr(K, "_config"):
        return "\n"

    cfg = K._config

    for k, v in cfg.items():
        env_str += "{}: {}\n".format(k, v)
    return env_str


def collect_env_info():
    data = []
    data.append(("sys.platform", sys.platform))
    data.append(("Python", sys.version.replace("\n", "")))
    data.append(("numpy", np.__version__))

    data.append(("Keras", keras.__version__))
    data.append(("Tensorflow", tf.__version__))

    try:
        import PIL

        data.append(("Pillow", PIL.__version__))
    except ImportError:
        pass

    try:
        import cv2

        data.append(("cv2", cv2.__version__))
    except ImportError:
        pass
    env_str = tabulate(data) + "\n"
    env_str += _pretty_print_keras_config(env_str)
    env_str += "CUDA Devices: {}".format(os.environ["CUDA_VISIBLE_DEVICES"])
    return env_str
