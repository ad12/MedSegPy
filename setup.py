#!/usr/bin/env python

import os
import sys
from os import path
from shutil import rmtree

from setuptools import Command, find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


class UploadCommand(Command):
    """Support setup.py upload.
    Adapted from https://github.com/robustness-gym/meerkat.
    """

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        os.system("twine upload dist/*")

        self.status("Pushing git tags…")
        os.system("git tag v{0}".format(get_version()))
        os.system("git push --tags")

        sys.exit()


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "medsegpy", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("MEDSEGPY_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


def get_required_packages():
    """Returns list of required packages based on tensorflow and keras versions."""
    default = [
        "Pillow",  # you can also use pillow-simd for better performance
        "matplotlib",
        "seaborn",
        "mock",
        "fvcore",
        "pydot",
        "pandas",
        "medpy",
        "numpy",
        "h5py",
        "natsort",
        "scipy",
        "scikit-image",
        "simpleitk",
        "configparser",
        "resnet",
        "more-itertools",
    ]
    return default


setup(
    name="medsegpy",
    version=get_version(),
    author="Arjun Desai",
    url="https://github.com/ad12/MedSegPy",
    description="MedSegPy is a framework for research on medical image segmentation.",
    packages=find_packages(exclude=("configs", "tests", "*.tests", "*.tests.*", "tests.*")),
    python_requires=">=3.6",
    install_requires=get_required_packages(),
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": [
            "flake8==4.0.1",
            "isort",
            "black==22.8.0",
            "flake8-bugbear",
            "flake8-comprehensions",
            "sphinx-rtd-theme",
            "nbsphinx",
            "recommonmark",
        ],
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # $ setup.py publish support.
    cmdclass={"upload": UploadCommand},
)
