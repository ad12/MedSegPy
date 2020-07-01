# Use Custom Datasets

If you want to use a custom dataset while also reusing medsegpy's data loaders,
you will need to

1. Optionally perform data augmentation
2. Store data in a medsegpy friendly way.
3. Register metadata for you dataset (i.e., tell medsegpy how to obtain your dataset).

Next, we explain the above two concepts in details.

### Data Augmentation
Volume-specific data augmentation can optionally be done outside of medsegpy.
If augmentations are used, define different augmentations or series of augmentations with a unique numeric identifier. This identifier will be the augmentation number for different scans (see below).

Note that augmentation should only be done on the training data. If augmentations are done on the validation and testing data, medsegpy functionality cannot be guaranteed.

### Data format
Currently, data can be stored as 2D slices or 3D volumes in the h5df format.

#### 2D Data
Many medical imaging modalities acquire single-slice acquisitions (CT, Xray, etc.).
Additionally, 3D volumes are often split into 2D slices when training 2D networks to
increase data speeds.

Data stored in the 2D format must follow a specific naming convention:
  * Subject id: 7 digits
  * Timepoint: 2 digits
  * Augmentation Number: 2 digits. Should be `00` for volumes that are not augmented
  * Slice number: 3 digits (1-indexed)

**Readable format:** `SubjectID_Timepoint-AugmentationNumber_SliceNumber`

**String format:** `%07d_V%02d-Aug%02d_%03d`

**Regex:** `([\d]+)_V([\d]+)-Aug([\d]+)_([\d]+)`

**Examples:**
- `0123456_V01-Aug00_001`: Subject 0123456, Timepoint 1, No Augmentation, Slice 1
- `0123456_V00-Aug00_001`: Subject 0123456, Timepoint 0,  No Augmentation, Slice 1
- `0123456_V00-Aug04_001`: Subject 0123456, Timepoint 0, Augmentation 4, Slice 1
- `0123456_V00-Aug00_999`: Subject 0123456, Timepoint 0, Augmentation 4, Slice 999

The augmentation number is used to keep track of what augmentations are done. 
When naming files, note that slices should start at slice 1.

###### Image files
Image files should end with a `.im` extension. The file should contain a dataset
`data`, which contains a `HxWx1` shaped array corresponding to the slice.

For example, `0123456_V00-Aug00_999.im` contains slice 999 from the volume
`0123456_V00-Aug00`.

###### Segmentation files
Ground truth masks should end with a `.seg` extension. The file should contain a
dataset `data`, which contains a `HxWx1xC` shaped binary array corresponding to the segmentation for the slice. Here, `C` refers to masks for different classes.

For example, `0123456_V00-Aug00_999.seg` contains segmentations for slice 999 from the volume `0123456_V00-Aug00`.

#### 3D Data
Data stored in the 3D format must follow a slightly different naming convention:
  * Subject id: 7 digits
  * Timepoint: 2 digits

**Readable format:** `SubjectID_Timepoint`

**String format:** `%07d_V%02d`

**Regex:** `([\d]+)_V([\d]+)`

Unlike the 2D files, where image and segmentation data are separated into different files,
3D data will have both in a single h5df file under different keys: `volume` for image data and `seg`
for segmentation masks.

We are working on adding the augmentation number (`Aug` flag) for parsing in 3D data!

#### Collating segmentations
Segmentations can also be collated (combined) to form segmentations for
superclasses. For example, if segmentations for "dog" and "cat" were stored
at index `0` and `2` in the segmentation file, to segment the both classes as a single class, specify the 
tuple `(0, 2)` as the index to segment.

#### How h5 files are read
Below are examples detailing the h5df structure for 2D and 3D data.

```python
import h5py

# ========= 2D Data =========
# Read slice 999 for volume 0123456_V00-Aug00.

# Read image slice.
with h5py.File("0123456_V00-Aug00_999.im") as f:
    image = f["data"][:]  # shape: HxWx1

# Read segmentations.
with h5py.File("0123456_V00-Aug00_999.seg") as f:
    mask = f["data"][:]  # shape: HxWx1xC
    
# ========= 3D Data =========
with h5py.File("0123456_V00.h5") as f:
    volume = f["volume"][:]  # shape: HxWxD
    mask = f["seg"][:]  # shape: HxWxDxC
```

#### Data paths
Data is often split into training, validation, and testing data. Each split
should be in a different directory. Image and segmentation files should be stored in the appropriate folder.

### Register Dataset
To let medsegpy know how to obtain a dataset named "my_dataset", you will impolement a function that
returns the items in your dataset and then tell medsegpy about this function

```python
def get_dicts():
    ...
    return list[dict] in the following format

from medsegpy.data import DatasetCatalog
DatasetCatalog.register("my_dataset", get_dicts)
```

Here, the snippet associates "my_dataset" with a function that returns the data. The registration
is effective as long as the process is running.

The function can process data from its original format into either one of the following:

1. MedSegPy's standard dataset dict, described below. This will work with many builtin features
in MedSegPy, so it is recommended when it is sufficient for your task.
2. Your custom dataset dict. You can choose to return arbitrary dicts that are designed to work
with your custom dataloader.

#### Standard Dataset Dicts
For standard semantic segmentation tasks, we load the original dataset into `list[dict]`.
Each dictionary is required to contain a set of keys. Because all data must currently be stored
as 2D slices in the h5 format (as described above), the following keys are required for all
dictionaries:

+ `file_name` (str): the full path to the image file for this slice.
+ `sem_seg_file` (str): the full path to the semantic segmentation file for this slice.
+ `scan_id` (str): the scan this slice belongs to
+ `slice_id` (int): The slice this file corresponds to. Should be 1-indexed, meaning the
first slice of every volume has `slice_id=1`.
+ `scan_num_slices`: the total number of slices in the scan volume.

The following keys are optional:
+ `subject_id` (int): the subject id corresponding to this scan
+ `time_point` (int): the time point at which the scan was acquired

All keys begining with `scan_` will be interpreted as special keys unique to the scan.
These will be returned as part of the input dictionary during inference.
In built-in medsegpy functions, these keys will also serve as override keys for any default
metadata associated with the dataset (see metadata section below). For example, if the dataset has a metadata key
`spacing`, the value for `spacing` is typically used for all elements in the dataset.
However, if a dataset dictionary has the key `scan_spacing`, the value of `scan_spacing` will
override the default metadata value.

MedSegPy will be expanding to support data stored in 3D soon. We will update
this section once that is complete.

#### Examples
For examples on registering datasets, see 
[datasets/oai.py](../modules/engine.html#medsegpy.data.datasets.oai)

### Register Metadata for a Dataset

To let medsegpy know how to obtain a dataset named "my_dataset", you will need
to add metadata for your specific dataset. Metadata names and types are shown
below.

Required:
+ `scan_root` (`str`): the directory path where images/segmentation files for the dataset are stored.
+ `category_ids` (sequence of `int` or `tuple[int]`): Category ids corresponding to different classes. Supports segmentation collating.
+ `categories` (sequence of `str`): Sequence of category names. 1-to-1 with `category_ids`.
+ `category_id_to_contiguous_id` (dict of `int/tuple[int]`->`int`): Maps
category ids to contiguous ids (0-indexed).
+ `evaluator_type` (`str`): value should be `"SemSegEvaluator"`

Optional:
+ `spacing` (tuple of `float`): the spacing in millimeters for scan volumes `(dH, dW, ...)`. Required for some segmentation metrics.
+ `category_abbreviations` (sequence of `str`): Abbreviations for categories.
1-to-1 with `categories`.
+ `category_colors` (sequence of `(R,G,B)`): R,G,B colors for different categories.

For data that is split into train/val/test splits, each split should be registered as a different dataset.

Below is an example for registering with training, validation, and testing splits. Segmentations for this data have 4 classes (in order): dog, human, cat, tree. We also want to collate the `dog` and `cat` categories into a new category `pet`.

```python
from medsegpy.data import MetadataCatalog
category_info = [
    {"id": 0, "name": "dog"},
    {"id": 1, "name": "human"},
    {"id": 2, "name": "cat"},
    {"id": 3, "name": "tree"},
    {"id": (0,2), "name": "pet"},  # collate "dog" & "cat" into "pet"
]
datasets_to_path = {
    "my_dataset_train": "/path/to/train/split",
    "my_dataset_val": "/path/to/val/split",
    "my_dataset_test": "/path/to/test/split",
}

for dataset_name, scan_root in datasets_to_path.items():
    MetadataCatalog.get(dataset_name).set(
        scan_root=scan_root,
        category_ids=[x["id"] for x in category_info],
        categories=[x["name"] for x in category_info],
        category_id_to_contiguous_id={
            x["id"]: idx for idx, x in enumerate(OAI_CATEGORIES)
        },
        evaluator_type="SemSegEvaluator",
    )
```

### Update the Config for New Datasets

Once you've registered the dataset, you can use the name of the dataset (e.g., "my_dataset" in
example above) in `{TRAIN,VAL,TEST}_DATASET`.
