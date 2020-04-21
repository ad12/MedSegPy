import os
import sys

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from tabulate import tabulate


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