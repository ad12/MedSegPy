# MedSegPy
MedSegPy is a framework for training segmentation networks on medical images.
It is designed to utilize the Keras/Tensorflow libraries and make training easily
configurable.

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
