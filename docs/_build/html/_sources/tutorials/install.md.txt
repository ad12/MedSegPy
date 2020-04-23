## Installation
We recommend using the Anaconda virtual environment to control package
visibility. For a detailed list of requirements, see
[environment.yml](environment.yml).

*Disclaimer*: The installation process has not been verified on cuda
versions >9.0.

### Requirements
- Linux or macOS with Python ≥ 3.6
- keras >=2.1.6,<2.2.0
- tensorflow-gpu >=1.8.0,<2.0.0
- graphviz (download via `conda install graphviz`)

Note that the TensorFlow version 1.8.0 is used by default and is
is prebuilt with cuda9.0. For other cuda versions, download
a suitable TensorFlow version. Tensorflow v1 binaries are not build
with cuda 10.1. To use cuda 10.1, you will have to build
tensorflow from source. If you do not have a gpu, replace the
`tensorflow-gpu` package with `tensorflow`.

### Build MedSegPy from Source
```bash
python -m pip install 'git+https://github.com/ad12/MedSegPy.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone (recommended):
git clone https://github.com/ad12/MedSegPy.git
cd MedSegPy && python -m pip install -e .
```

You often need to rebuild medsegpy after reinstalling TensorFlow and/or Keras.
