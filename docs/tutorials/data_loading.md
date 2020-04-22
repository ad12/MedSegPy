
# Use Custom Dataloaders

*Note*: The current data loading procedure is currently under refactoring.
While the process below is currently supported, it is subject to change.

## How the Existing Dataloader Works

MedSegPy contains a builtin data loading pipeline.
It's good to understand how it works, in case you need to write a custom one.

MedSegPy provides an interface for loading and structuring data stored in
different ways (3D volumes, 2D slices, etc.). Data structuring consists of
scattering a single element into multiple element (3D volume -> 3D blocks) or
gathering multiple elements into a single element
(multiple 2D slices -> 3D volume). For example, if data from a 3D scan is
saved slice-wise across different h5 files and we want to train using a
3D network, we can use MedSegPy's interface for gathering data from different
files into a single volume.

MedSegPy's loading/structuring interface is defined by the
[`Generator`](../modules/data.html#detectron2.data.Generator) abstract class.
Subclasses determine how the data is loaded/structured. The generator implements
two generator methods `img_generator` and `img_generator_test` to be used for
training and testing, respectively.

`img_generator` (used with [`model.fit_generator`](https://keras.io/models/sequential/)):
1. It takes the
   [GeneratorState](../modules/data.html#detectron2.data.GeneratorState) and
   loads the registered dataset for that state. The dataset is a loaded as 2
   lists corresponding to the h5 files for the input and the masks.
   Details about the dataset format and dataset registration can be found in
   [datasets](datasets.html).
2. Inputs/masks are read from h5 files and structured. Each generator
   structures data differently. This means that data stored in
   different ways (slice-wise, 3D, etc.) must be used with specific generators.
3. The inputs/outputs are batched into an ndarray of dimensions `Nx...`, where
   `N` is the batch size.

`img_generator_test`:
1. It takes the trained model and loads the registered dataset for the
   `GeneratorState.TESTING` state.
2. Batches data by `scan_id`. Data corresponding to the same `scan_id` will be
   processed and structured into a 3D volume of shape `DxHxW`.
3. Runs `model.predict` on data for a specific `scan_id`.
4. Yields `x` (input), `y_true` (ground truth masks), `y_pred` (predicted mask)
   `scan_id`, `time_elapsed` (time to run prediction).

## Dataloader example
Below we describe loading data and training a model using the for
OAI iMorphics 2D dataset, a dataset where data is stored
as 2D slices. For more information on acceptable dataset h5 files, see
[datasets](datasets.html).

```python
from medsegpy.config import UNetConfig
from medsegpy.data import OAIGenerator, GeneratorState
from medsegpy.modeling import get_model

cfg = UNetConfig()
cfg.TRAIN_DATASET = "oai_2d_train"
cfg.VAL_DATASET = "oai_2d_val"
cfg.TEST_DATASET = "oai_2d_test"

model = get_model(cfg)
model.compile(...)  # compile with optimizer, loss, metrics, etc.

generator = OAIGenerator(cfg)
train_steps, val_steps = generator.num_steps()

# Start training
model.fit_generator(
    generator=generator.img_generator(GeneratorState.TRAINING),
    steps_per_epoch=train_steps,
    validation_data=generator.img_generator(GeneratorState.VALIDATION),
    validation_steps=val_steps,
    ...  # other params for fit_generator
)

# Run inference.
for x, y_true, y_pred, scan_id, time_elapsed in generator.img_generator_test(model):
    # x (ndarray): scan volume - shape DxHxW
    # y_true (ndarray): ground truth masks - shape DxHxWxC
    # y_pred (ndarray): predictions (probabilities) - DxHxWxC
    # scan_id (str): scan id
    # time_elapsed (float): Time for segmenting whole volume.

    # Do inference related things.
```

## Write a Custom Dataloader
See the [abdominal CT generator](../modules/data.html#medsegpy.data.CTGenerator)
for an example.

## Use a Custom Dataloader

If you use [DefaultTrainer](../modules/engine.html#medsegpy.engine.defaults.DefaultTrainer),
you can overwrite its `_build_data_loaders` and `build_test_data_loader` methods to use your own dataloader.
See the [abCT training](../../tools/ct_train.py)
for an example.

If you write your own training loop, you can plug in your data loader easily.
