# Use Models

Models (and their sub-models) in medsegpy are built by
functions such as `build_model`:
```python
from medsegpy.modeling.meta_arch import build_model
model = build_model(cfg)  # returns a medsegpy.modeling.Model
```

Note that `build_model` only builds the model structure, and fills it with random parameters.
To load an existing checkpoint to the model, set
`cfg.INIT_WEIGHTS` to the appropriate weights file.
MedSegPy recognizes models in Keras's `.h5` format.

You can use a model by just `outputs = model.predict(inputs)`.
Next, we explain the inputs/outputs format used by the builtin models in MedSegPy.

For a detailed list of models see 
[modeling/meta_arch](../modules/modeling.html#medsegpy.modeling.meta_arch)

### Making a Custom Model
MedSegPy is designed to support custom models and is built so that they can easily be integrated into
the current structure.

All models must extend the MedSegPy
[Model](../modules/modeling.html#medsegpy.modeling.model.Model) interface. This interface has a builtin
method that makes testing on different scans and running inference relatively simple.

Each model is associated with a unique config type (see the [config tutorial](configs.html).
Here you will define fields that are specific to controlling properties of your model architecture.

If your model is very similar to existing models, see if you can modify existing configs to include
a handful of fields that can be used to control your additions. If you do, make sure to turn 
those options off by default so as to not interfere with expected default functionality. If your 
model behaves similarly to existing models but requires some pretty extensive additions, we recommend
extending/subclassing your config from the existing config corresponding to the similar model.
