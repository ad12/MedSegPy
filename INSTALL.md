## Installation
Download this repository to your disk. The path to this repository should not
have any spaces. In general, this library does not handle folder paths that
have spaces in between folder names.

We recommend using the Anaconda virtual environment to control package
visibility. For a detailed list of requirements, see
[environment.yml](environment.yml).

Note that the TensorFlow version
in this file is only compatible with cuda9.0. For other cuda versions, download
a suitable TensorFlow version. If you do not have a gpu, replace the
`tensorflow-gpu` package with `tensorflow`.

*Disclaimer*: The installation process has not been verified on cuda
versions >9.0.

### Download Packages
```bash
# ========= With conda & cuda9.0 =========
conda create env -f environment.yml

# ========= Individual libraries =========
conda create env -n medsegpy_env python=3.6
conda activate medsegpy_env
pip install cython

# Change versions based on compatibility with cuda version.
# Keras > 2.2.0 may not be compatible.
pip install tensorflow-gpu==1.8.0 keras==2.1.6

pip install matplotlib seaborn
pip install configparser h5py natsort pandas scipy scikit-image medpy simpleitk
pip install graphviz pydot
```
