
# Use Custom Dataloaders

## How the Existing Dataloader Works

MedSegPy contains a builtin data loading pipeline.
It's good to understand how it works, in case you need to write a custom one.

MedSegPy provides an interface for loading and structuring data stored in
different ways (3D volumes, 2D slices, etc.). Data structuring consists of
scattering a single element into multiple elements (3D volume -> 2D/3D patches) or
gathering multiple elements into a single element
(multiple 2D slices -> 3D volume). For example, if data from a 3D scan is
saved slice-wise across different h5 files and we want to train using a
3D network, we can use MedSegPy's interface for gathering data from different
files into a single volume (note this functionality is still being built).

MedSegPy's loading/structuring interface is defined by the
[`DataLoader`](../modules/data.html#medsegpy.data.data_loader.DataLoader) abstract class.
This class extends the keras 
[`Sequence`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/keras/utils/Sequence)
class. Like Sequences, `DataLoaders` implement a `__getitem__` method that can be used for fetching
batches. For training and validation purposes, we recommend following the keras API for loading
data with sequences.

As mentioned above, medical data often requires structuring/patching. This can result in returning batches
of elements that are subsets of a single scan. For example, a data loader that indexes over 2D slices
of a 3D scan is incredibly useful for training 2D models. However, during inference, metrics are 
typically calculated per scan and restructuring data outside of the data loader can be difficult.

To simplify inference and downstream metric calculation, each data loader implements an
`inference` method, which takes in a medsegpy 
[`Model`](../modules/modeling.html#medsegpy.modeling.model.Model) and keyword arguments that 
are typically used with [`predict_generator`](https://keras.io/models/sequential/#predict_generator). 
In `inference`, the data loader does the following:
1. It loads all dataset dictionaries corresponding to a given scan
2. Structures data in these dictionaries based on the data loader's defined structuring method.
3. Runs inference on scan data
4. Reformats scan data. Images/volumes will be of the shape `HxWx...`. Semantic segmentation
masks and predictions will have shape `HxWx...xC`.
5. Yields a dictionary of inputs and outputs

This method continues to yield input and output data in the medsegpy format until data for all
scans are yielded. For more information, see 
[DataLoader](../modules/data.html#medsegpy.data.DataLoader).

## Dataloader example
Below we describe loading data and training a model using the for
OAI iMorphics 2D dataset, a dataset where 3D volumes are stored
as 2D slices. For more information on acceptable dataset h5 files, see
[datasets](datasets.html).

The `DefaultDataLoader` handles both 2D single-slice scans
and 3D scans stored as 2D slices. For more information on other dataloaders,
see data loaders in [medsegpy.data.data_loader](../modules/data.html#medsegpy.data.DataLoader).

```python
from medsegpy.config import UNetConfig
from medsegpy.data import build_loader, DatasetCatalog, DefaultDataLoader
from medsegpy.modeling import get_model

cfg = UNetConfig()
cfg.TAG = "DefaultDataLoader"  # Specify the data loader type
cfg.TRAIN_DATASET = "oai_2d_train"
cfg.VAL_DATASET = "oai_2d_val"
cfg.TEST_DATASET = "oai_2d_test"

model = get_model(cfg)
model.compile(...)  # compile with optimizer, loss, metrics, etc.

# Using built-in methods to create loaders.
# To build them from scratch, see implementation
# of `build_loader`.
train_loader = build_loader(
    cfg, 
    cfg.TRAIN_DATASET, 
    batch_size=10,
    is_test=False,
    shuffle=True,
    drop_last=True,
)
val_loader = build_loader(
    cfg, 
    cfg.VAL_DATASET, 
    batch_size=10,
    is_test=False,
    shuffle=True,
    drop_last=True,
)
test_loader = build_loader(
    cfg, 
    cfg.TEST_DATASET, 
    batch_size=10,
    is_test=False,
    shuffle=True,
    drop_last=False,
)

# Start training
model.fit_generator(
    train_loader,
    validation_data=val_loader,
    ...
)

# Run inference.
for input, output in test_loader.inference(model):
    # Do inference related things.
```

## Write a Custom Dataloader
Coming soon!

## Use a Custom Dataloader

If you use [DefaultTrainer](../modules/engine.html#medsegpy.engine.trainer.DefaultTrainer),
you can overwrite its `_build_data_loaders` and `build_test_data_loader` methods to use your own dataloader.

If you write your own training loop, you can also plug in your data loader easily.
