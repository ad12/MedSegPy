import os
import re
import socket
import warnings
from typing import Sequence, Union

import yaml
from fvcore.common.file_io import PathManager

from medsegpy.utils.env import settings_dir

# Path to the repository directory.
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class Cluster:
    """Tracks config of different nodes/clusters.

    This class is helpful for managing different storage paths across different
    nodes/clusters without the overhead of duplicating the codebase across
    multiple nodes.

    To identify the current node, we inspect the hostname.
    This can be problematic if two machines have the same hostname, though
    this has not been an issue as of yet.

    DO NOT use the node's public ip address to identify it. Not only is this not
    returned by ``socket.hostname()``, but there are also some security issues.

    Note:
        This class is not thread safe. Saving/deleting configs should be done on
        the main thread.
    """

    def __init__(
        self,
        name: str,
        patterns: Union[str, Sequence[str]],
        data_dir: str = None,
        results_dir: str = None,
    ):
        """
        Args:
            name (str): The name of the cluster. Name is case-sensitive.
            patterns (Sequence[str]): Regex pattern(s) for identifying cluster.
                Cluster will be identified by
                ``any(re.match(p, socket.gethostname()) for p in patterns)``.
            data_dir (str, optional): The data directory. Defaults to
                ``os.environ['MEDSEGPY_RESULTS']`` or ``"./datasets"``.
            results_dir (str, optional): The results directory. Defaults to
                `"os.environ['MEDSEGPY_DATASETS']"` or ``"./results"``.
        """
        self.name = name

        if isinstance(patterns, str):
            patterns = patterns
        self.patterns = patterns

        self._data_dir = data_dir
        self._results_dir = results_dir

    @property
    def data_dir(self):
        path = self._data_dir
        if not path:
            path = os.environ.get("MEDSEGPY_DATASETS", "./datasets")
        return PathManager.get_local_path(path)

    @property
    def results_dir(self):
        path = self._results_dir
        if not path:
            path = os.environ.get("MEDSEGPY_RESULTS", "./results")
        return PathManager.get_local_path(path)

    def save(self):
        """Save cluster config to yaml file."""
        filepath = self.filepath()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            data = {(k[1:] if k.startswith("_") else k): v for k, v in self.__dict__.items()}
            yaml.safe_dump(data, f)

    def delete(self):
        """Deletes the config file for this cluster."""
        filepath = self.filepath()
        if os.path.isfile(filepath):
            os.remove(filepath)

    def filepath(self):
        """Returns config file path.

        Note:
            This does not guarantee the config exists. To save the cluster config to a file,
            use `save()`.

        Returns:
            str: The config file path.
        """
        return os.path.join(self._config_dir(), f"{self.name}.yaml")

    @property
    def save_dir(self):
        """Deprecated: Legacy alias for `self.results_dir`"""
        warnings.warn(
            "`save_dir` is deprecated and will be removed in v0.0.2. Use `results_dir` instead",
            FutureWarning,
        )
        return self.results_dir

    @classmethod
    def all_clusters(cls):
        config_dir = cls._config_dir()
        clusters = []
        if os.path.isdir(config_dir):
            files = sorted(os.listdir(config_dir))
            for f in files:
                clusters.append(cls.from_config(os.path.join(config_dir, f)))
        return clusters

    @classmethod
    def cluster(cls):
        """Searches saved clusters by regex matching with hostname.

        Note:
            The cluster must have been saved to a config file. Also, if
            there are multiple cluster matches, only the first (sorted alphabetically)
            will be returned.

        Returns:
            Cluster: The current cluster.
        """
        clusters = cls.all_clusters()
        hostname = socket.gethostname()
        for clus in clusters:
            if any(re.match(p, hostname) for p in clus.patterns):
                return clus
        return _UNKNOWN

    @classmethod
    def from_config(cls, name):
        """
        Args:
            name (str): Cluster name or path to config file.
        Returns:
            Cluster: The Cluster object
        """
        if not os.path.isfile(name):
            filepath = os.path.join(cls._config_dir(), f"{name}.yaml")
        else:
            filepath = name
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Config file not found: {name}")

        with open(filepath, "r") as f:
            cfg = yaml.safe_load(f)
        return cls(**cfg)

    @staticmethod
    def _config_dir():
        return os.path.join(settings_dir(), "clusters")

    @staticmethod
    def working_cluster() -> "Cluster":
        return _CLUSTER

    @staticmethod
    def set_working_cluster(cluster=None):
        """Sets the working cluster.

        Args:
            cluster (`str` or `Cluster`): The cluster name or cluster.
                If ``None``, will reset cluster to _UNKNOWN, meaning default
                data and results dirs will be used.
        """
        set_cluster(cluster)

    def __repr__(self):
        return "Cluster({})".format(
            ", ".join("{}={}".format(k, v) for k, v in self.__dict__.items())
        )


def set_cluster(cluster: Union[str, Cluster] = None):
    """Sets the working cluster.

    Args:
        cluster (`str` or `Cluster`): The cluster name or cluster.
            If ``None``, will reset cluster to _UNKNOWN, meaning default
            data and results dirs will be used.
    """
    if cluster is None:
        cluster = _UNKNOWN
    elif isinstance(cluster, str):
        if cluster.lower() == _UNKNOWN.name.lower():
            cluster = _UNKNOWN
        else:
            cluster = Cluster.from_config(cluster)
    global _CLUSTER
    _CLUSTER = cluster


_UNKNOWN = Cluster("UNKNOWN", [])  # Unknown cluster
_CLUSTER = Cluster.cluster()  # Working cluster
