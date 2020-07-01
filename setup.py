#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from os import path
from setuptools import find_packages, setup

import keras
import tensorflow as tf

tf_ver = [int(x) for x in tf.__version__.split(".")[:2]]
assert [1, 8] <= tf_ver < [2, 0], "Requires TensorFlow >=1.8,<2.0"
keras_ver = [int(x) for x in keras.__version__.split(".")[:3]]
assert [2, 1, 6] <= keras_ver < [2, 2, 0], "Requires Keras >=2.1.6, <2.2.0"


def get_version():
    init_py_path = path.join(
        path.abspath(path.dirname(__file__)), "medsegpy", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][
        0
    ]
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


setup(
    name="medsegpy",
    version=get_version(),
    author="Arjun Desai",
    url="https://github.com/ad12/MedSegPy",
    description="MedSegPy is a framework for research on medical image "
    "segmentation.",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    install_requires=[
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
    ],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": [
            "flake8",
            "isort",
            "black==19.3b0",
            "flake8-bugbear",
            "flake8-comprehensions",
        ],
    },
)
