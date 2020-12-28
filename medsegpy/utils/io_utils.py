import os
import pickle
import re
from abc import ABC
from typing import Any

import h5py
from fvcore.common.file_io import PathHandler, PathManager

from medsegpy.utils.cluster import Cluster


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


def format_exp_version(dir_path, new_version=True, mkdirs=False, force=False):
    """Adds experiment version to the directory path. Returns local path.

    If `os.path.basename(dir_path)` starts with 'version', assume the version
    has already been formatted.

    Args:
        dir_path (str): The directory path corresponding to the version.
        force (bool, optional): If `True` force adds version even if 'version'
            is part of basename.

    Returns:
        str: The formatted dirpath
    """
    dir_path = PathManager.get_local_path(dir_path)
    if not os.path.isdir(dir_path):
        return os.path.join(dir_path, "version_001")
    if not force and re.match("^version_[0-9]*", os.path.basename(dir_path)):
        return dir_path
    version_dir, version_num = _find_latest_version_dir(dir_path)
    if new_version:
        version_num += 1
        version_dir = f"version_{version_num:03d}"
    version_dirpath = os.path.join(dir_path, version_dir)
    if mkdirs:
        PathManager.mkdirs(version_dirpath)
    return version_dirpath


def _find_latest_version_dir(dir_path):
    version_dirs = [
        (x, int(x.split("_")[1]))
        for x in os.listdir(dir_path) if re.match("^version_[0-9]*", x)
    ]
    if len(version_dirs) == 0:
        version_dir, version_num = None, 0
    else:
        version_dirs = sorted(version_dirs, key=lambda x: x[1])
        version_dir, version_num = version_dirs[-1]
    return version_dir, version_num


class GeneralPathHandler(PathHandler, ABC):
    PREFIX = ""

    def _get_supported_prefixes(self):
        return [self.PREFIX]

    def _get_local_path(self, path: str, **kwargs: Any):
        name = path[len(self.PREFIX) :]
        return os.path.join(Cluster.working_cluster().results_dir, self._project_name(), name)

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
