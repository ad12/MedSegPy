# MRSegPy
MRSegPy is a command-line-enabled framework for training segmentation networks on magnetic resonance (MR) images. It is designed to utilize the Keras/Tensorflow libraries and abstractly handle training hyperparameters,
such as loss functions, early stopping, adaptive learning rates, etc.

## Getting Started
### Installation
Download this repository to your disk. Note that the path to this repo should not have any spaces. In general, this library does not handle folder paths that have spaces in between folder names.

#### Virtual Environment
We recommend using the Anaconda virtual environment to run python. If Anaconda is not installed, please do so from [Anaconda Distribution](https://www.anaconda.com/distribution/).

An `environment.yml` file is provided in this repo containing all libraries used. To install all dependencies in a new environment, follow the steps below:
1. Open Terminal/Bash
2. Navigate to folder containing `environment.yml` file
3. Run `conda env create -f environment.yml`

### Formatting Data
All data should be written to h5 files slice-wise, meaning each slice is its own file. Each file should have the following format:

**Readable format:** `PatientID_Timepoint-AugmentationNumber_SliceNumber`

**String format:** `%07d_V01-Aug%02d_%03d`

**Regex:** `([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)`

**Examples:**
- `0123456_V01-Aug00_001`: Patient 0123456, Timepoint 1, Augmentation None, Slice 1
- `0123456_V00-Aug00_001`: Patient 0123456, Timepoint 0, Augmentation None, Slice 1
- `0123456_V00-Aug04_001`: Patient 0123456, Timepoint 0, Augmentation 4, Slice 1
- `0123456_V00-Aug00_999`: Patient 0123456, Timepoint 0, Augmentation 4, Slice 999

Data should be stored in 3 directories corressponding to training/validation/testing sets respectively (preferred) or stored in a single directory (if cross validation is being used).

## Training Networks
### Architectures
MRSegPy offers several state-of-the-art segmentation architectures, thanks much in part to the open-source community. These include, but are not limited to, [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28), [SegNet](https://ieeexplore.ieee.org/abstract/document/7803544), [DeeplabV3+](https://arxiv.org/abs/1802.02611).

Each of these architectures has a corresponding configuration that allows for easy programming of the architecture and training protocol.

### Starting Training
All training experiments can be accessed via the `nn_train` module from the command line. The module takes a sub-argument specifying which architecture to train. Each sub-argument exposes the parameters in the configuration and will be detailed in the help menu.

For help, run `python -m nn_train -h`, which will show you the names of the different supported architectures. Some of these architectures are listed below

```
unet_2d, segnet_2d, deeplabv3_2d, res_unet, anisotropic_unet, refinenet, unet_3d, unet_2_5d
```

### Supported Training Protocols
#### K-fold Cross-validation
Cross-validation is often suggested when statistical power of a holdout test set is too low (i.e. limited testing data). MRSegPy supports subject-wise k-fold cross validation, where all subjects in the training set are split into k different bins. Note that we do not bin data by slice or by scan because having scans from the same subject overlap in training/validation/testing bins does not fulfill the disjoint protocol during training.

```
```
