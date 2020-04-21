import os
import numpy as np
import sys
from tabulate import tabulate

import keras
import keras.backend as K
import tensorflow as tf

__all__ = ["collect_env_info"]


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
    env_str += K._config + "\n"
    env_str += "CUDA Devices: {}".format(os["CUDA_VISIBLE_DEVICES"])
    return env_str

