# Use Models

Models (and their sub-models) in medsegpy are built by
functions such as `get_model`:
```python
from medsegpy.modeling import build_model
model = build_model(cfg)  # returns a torch.nn.Module
```

Note that `build_model` only builds the model structure, and fill it with random parameters.
To load an existing checkpoint to the model, set
`cfg.INIT_WEIGHTS` to the appropriate weights file.
MedSegPy recognizes models in Keras's `.h5` format.

You can use a model by just `outputs = model.predict(inputs)`.
Next, we explain the inputs/outputs format used by the builtin models in MedSegPy.

#### Models and Generators

The [Generator.img_generator]( ../modules/data.html#medsegpy.data.Generator.img_generator) yields batches of input and ground truth masks in the expected Keras format. See [data_loading](data_loading.html)] for an example of how this is done.
