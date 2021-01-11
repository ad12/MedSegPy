"""Overload Keras deserialization for default with custom layers and model names."""
from keras.layers import deserialize as _deserialize


from medsegpy.utils import env

from .pooling import MaxPoolingWithArgmax2D, MaxUnpooling2D
from .upsampling import BilinearUpsampling

if env.is_tf2():
    from tensorflow.python.keras.engine.functional import Functional
else:
    Functional = None


def deserialize(config, custom_objects=None):
    """Overload keras.layers.deserialize with MedSegPy builtin layers.

    Args:
      config: dict of the form {'class_name': str, 'config': dict}
      custom_objects: dict mapping class names (or function names)
          of custom (non-Keras) objects to class/functions

    Returns:
      Layer instance (may be Model, Sequential, Network, Layer...)
    """
    # Prevent circular dependencies.
    from ..model import Model  # noqa

    all_custom_objects = {
        "Model": Model,
        "BilinearUpsampling": BilinearUpsampling,
        "MaxPoolingWithArgmax2D": MaxPoolingWithArgmax2D,
        "MaxUnpooling2D": MaxUnpooling2D,
    }
    all_custom_objects.update(custom_objects)
    model = _deserialize(config, custom_objects=all_custom_objects)
    # if env.is_tf2() and isinstance(model, Functional):
    #     model = Model(model)
    return model
