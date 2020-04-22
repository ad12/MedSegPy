## Getting Started with MedSegPy
This document provides a brief intro of the usage of builtin command-line tools in medsegpy.

More detailed documentation coming soon.

### Adding a Custom Dataset
#### Data Format
To use the default dataloaders with a custom dataset, each file name should have the following format:

**Readable format:** `SubjectID_Timepoint-AugmentationNumber_SliceNumber`

**String format:** `%07d_V%02d-Aug%02d_%03d`

**Regex:** `([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)`

**Examples:**
- `0123456_V01-Aug00_001`: Subject 0123456, Timepoint 1, Augmentation None, Slice 1
- `0123456_V00-Aug00_001`: Subject 0123456, Timepoint 0, Augmentation None, Slice 1
- `0123456_V00-Aug04_001`: Subject 0123456, Timepoint 0, Augmentation 4, Slice 1
- `0123456_V00-Aug00_999`: Subject 0123456, Timepoint 0, Augmentation 4, Slice 999

There should be 2 [h5df](http://docs.h5py.org/en/stable/) files that correspond to each slice: an image file (`.im` extension) and a segmentation file (`.seg` extension). The image file should contain a dataset `data` with a `HxWx1` matrix corresponding to the slice.
The segmentation file should contain a dataset `data` with a `HxWx1xC` matrix corresponding
to the binary segmentations for the `C` different classes.

If data does not match this format, you will have to write your own data loader.

#### Declaring data splits
Data should be stored in 3 directories corresponding to training/validation/testing sets respectively (preferred). The 

## Training Networks
### Architectures
MRSegPy offers several state-of-the-art segmentation architectures, thanks in part to the open-source community.
These include, but are not limited to [U-Net](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28),
[SegNet](https://ieeexplore.ieee.org/abstract/document/7803544), and [DeeplabV3+](https://arxiv.org/abs/1802.02611).

Each of these architectures has a corresponding configuration that allows for easy programming of the architecture and
training protocol.

### Starting Training
All training experiments can be accessed via the `nn_train` module from the command line.
The module takes a sub-argument specifying which architecture to train.
Each sub-argument exposes the parameters in the configuration and will be detailed in the help menu.

For help, run `python -m nn_train -h`, which will show you the names of the different supported architectures.
Some of these architectures are listed below.

```bash
unet_2d, segnet_2d, deeplabv3_2d, res_unet, anisotropic_unet, refinenet, unet_3d, unet_2_5d
```

### Supported Training Protocols
#### K-fold Cross-validation
Cross-validation is often suggested when statistical power of a holdout test set is too low (i.e. limited testing data). MRSegPy supports subject-wise k-fold cross validation, where all subjects in the training set are split into k different bins. Note that we do not bin data by slice or by scan because having scans from the same subject overlap in training/validation/testing bins does not fulfill the disjoint protocol during training.
