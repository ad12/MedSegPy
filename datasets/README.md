# Setup Default Datasets

## Download
The default dataset included in the MedSegPy library is the OAI imorphics
dataset. It can be officially requested from [here](https://oai.epi-ucsf.org/datarelease/iMorphics.asp). However, by default the segmentation masks are not all ordered accurately, and the train/val/test split is not standardized. To download a polished version of this dataset with standardized splits, email `arjundd at stanford.edu`.

## Formatting dataset
Follow instructions on [dataset.md](../docs/tutorials/datasets) to set up the data. By default, no augmentations were performed on the OAI iMorphics dataset.

Setup scripts coming soon!

## Change dataset paths
Because of differences in directory paths, some minimal changes have to be made.

At the beginning of your training script, refresh the paths to the training directories.

```python
from medsegpy.data import MetadataCatalog

MetadataCatalog.get("oai_2d_train").scan_root = ""  # Path to train split
MetadataCatalog.get("oai_2d_val").scan_root = ""  # Path to val split
MetadataCatalog.get("oai_2d_test").scan_root = ""  # Path to test split
```
