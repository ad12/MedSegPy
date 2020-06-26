import getpass
import os
import re
import socket
from enum import Enum
from typing import List

# Path to the repository directory.
_REPO_DIR = os.path.join(os.path.dirname(__file__), "../..")


class Cluster(Enum):
    """Hacky way to keep track of the cluster you are working on.

    To identify the cluster, we inspect the hostname.
    This can be problematic if two clusters have the same hostname, though
    this has not been an issue as of yet.

    DO NOT use the machine's public ip address to identify it. While this is
    definitely more robust, there are security issues associated with this.

    Useful for developing with multiple people working on same and different
    machines.

    TODO (arjundd): make the paths configurable via experiments/preferences.
    """

    UNKNOWN = 0, []
    ROMA = 1, ["roma"]
    VIGATA = 2, ["vigata"]
    NERO = 3, ["slurm-gpu-compute.*"]

    def __new__(cls, value: int, patterns: List[str]):
        """
        Args:
            value (int): Unique integer value.
            patterns (`List[str]`): List of regex patterns that would match the
                hostname on the compute cluster. There can be multiple hostnames
                per compute cluster because of the different nodes.
            save_dir (str): Directory to save data to.
        """
        obj = object.__new__(cls)
        obj._value_ = value

        obj.patterns = patterns
        obj.dir_map = {}

        return obj

    @classmethod
    def cluster(cls):
        hostname = socket.gethostname()

        for clus in cls:
            for p in clus.patterns:
                if re.match(p, hostname):
                    return clus

        return cls.UNKNOWN

    def register_user(
        self, user_id: str, results_dir: str = "",
    ):
        """Register user preferences for paths.

        Args:
            user_id (str): User id found on the machine.
            data_dir (str): Default data directory.
                Paths starting with "data://" will be formated to this
                directory as the root. For example if `data_dir=/my/path`,
                then file path "data://data1" will be "/my/path/data1".
            results_dir (str): Default results directory.
                Performance is like that of data_dir expect with "results://"
                prefix.
        """
        if not results_dir:
            results_dir = os.path.abspath(os.path.join(_REPO_DIR, "results"))

        self.dir_map[user_id] = {
            "results_dir": results_dir,
        }

    @property
    def save_dir(self):
        user_id = getpass.getuser()
        if user_id not in self.dir_map:
            raise ValueError("User {} is not registered".format(user_id))
        return self.dir_map[user_id]["results_dir"]


# Environment variable for the current cluster that is being used.
CLUSTER = Cluster.cluster()

# Define path to your results folder.
_USER_PATHS = {
    "arjundd": {
        CLUSTER.ROMA: "/bmrNAS/people/arjun/results",
        CLUSTER.VIGATA: "/bmrNAS/people/arjun/results",
        CLUSTER.NERO: "/share/pi/bah/arjundd/results",
    },
    # New users add path preference below.
}

# Register default user paths.
_USER = getpass.getuser()
if _USER in _USER_PATHS:
    for cluster, results_dir in _USER_PATHS[_USER].items():
        cluster.register_user(_USER, results_dir)
