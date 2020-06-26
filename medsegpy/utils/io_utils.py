import os
import pickle
from abc import ABC
from typing import Any

import h5py
from fvcore.common.file_io import PathHandler, PathManager

from .cluster import CLUSTER


def load_h5(file_path):
    """Load data in h5df format.

    Args:
        file_path (str): Path to h5 file

    Returns:
        dict: dictionary representation of h5df data.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError("%s does not exist" % file_path)

    data = dict()
    with h5py.File(file_path, "r") as f:
        for key in f.keys():
            data[key] = f.get(key).value

    return data


def save_optimizer(optimizer, dirpath: str):
    """Serialize a model and add the config of the optimizer.

    Args:
        optimizer (keras.Optimzer): a Keras optimizer
        dirpath (str): Path to directory
    """
    if optimizer is None:
        return

    config = dict()
    config["optimizer"] = optimizer.get_config()

    filepath = os.path.join(dirpath, "optimizer.dat")
    # Save optimizer state
    save_pik(config, filepath)


def load_optimizer(dirpath: str):
    """Return model and optimizer in previous state.
    """
    from keras import optimizers

    filepath = os.path.join(dirpath, "optimizer.dat")
    model_dict = load_pik(filepath)
    optimizer_params = {k: v for k, v in model_dict.get("optimizer").items()}
    optimizer = optimizers.get(optimizer_params)

    return optimizer


def save_pik(data, filepath):
    """Save data using pickle.
    """
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_pik(filepath):
    """
    Load data using pickle
    :param filepath: filepath to load from
    :return: data saved using save_pik
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


class GeneralPathHandler(PathHandler, ABC):
    PREFIX = ""

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs: Any):
        name = path[len(self.PREFIX) :]
        return os.path.join(CLUSTER.save_dir, self._project_name(), name)

    def _open(self, path, mode="r", **kwargs):
        return PathManager.open(self._get_local_path(path), mode, **kwargs)

    def _mkdirs(self, path: str, **kwargs: Any):
        os.makedirs(self._get_local_path(path), exist_ok=True)

    def _project_name(self):
        return self.PREFIX[:-3]


class TechConsiderationsHandler(GeneralPathHandler):
    PREFIX = "tcv3://"

    def _project_name(self):
        return "tech-considerations"

PathManager.register_handler(TechConsiderationsHandler())
