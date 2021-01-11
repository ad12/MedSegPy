# Setup Default Datasets

## Download
The default dataset included in the MedSegPy library is the OAI imorphics
dataset. It can be officially requested from
[here](https://oai.epi-ucsf.org/datarelease/iMorphics.asp). However, by default
the segmentation masks are not all ordered accurately, and the train/val/test
split is not standardized. To download a polished version of this dataset with
standardized splits, email `arjundd at stanford.edu`.

## Formatting dataset
Follow instructions on [dataset.md](../docs/tutorials/datasets) to set up the
data. By default, no augmentations were performed on the OAI iMorphics dataset.

Setup scripts coming soon!

## Change dataset paths
If the machine/cluster you are working on is already supported, you do not have
to format any dataset paths.

If you added a new cluster, some minimal changes have to be made. For OAI iMorphics datasets, add the data paths on your cluster (see
[medsegpy/data/datasets/oai.py](medsegpy/data/datasets/oai.py)).
