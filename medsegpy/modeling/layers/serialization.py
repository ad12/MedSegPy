"""Overload Keras deserialization for default with custom layers and model names."""
from keras.layers import deserialize as _deserialize

from .pooling import MaxPoolingWithArgmax2D, MaxUnpooling2D
from .upsampling import BilinearUpsampling


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
    return _deserialize(config, custom_objects=all_custom_objects)
