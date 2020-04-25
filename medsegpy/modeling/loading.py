"""Model loading utils.

Adapted from `keras.saving.model_config`.
https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/saving/model_config.py#L55
"""
import json

try:
    import yaml  # noqa: E402
except ImportError:
    yaml = None


def model_from_config(config, custom_objects=None):
    """Instantiates a Keras model from its config.
    Arguments:
      config: Configuration dictionary.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
    Returns:
      A Keras model instance (uncompiled).
    Raises:
      TypeError: if `config` is not a dictionary.
    """
    if isinstance(config, list):
    raise TypeError('`model_from_config` expects a dictionary, not a list. '
                    'Maybe you meant to use '
                    '`Sequential.from_config(config)`?')
    from .layers import deserialize  # noqa
    return deserialize(config, custom_objects=custom_objects)


def model_from_yaml(yaml_string, custom_objects=None):
    """Parses a yaml model configuration file and returns a model instance.
    Arguments:
      yaml_string: YAML string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
    Returns:
      A Keras model instance (uncompiled).
    Raises:
      ImportError: if yaml module is not found.
    """
    if yaml is None:
    raise ImportError('Requires yaml module installed (`pip install pyyaml`).')
    config = yaml.load(yaml_string)
    from .layers import deserialize  # noqa
    return deserialize(config, custom_objects=custom_objects)


def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration file and returns a model instance.
    Arguments:
      json_string: JSON string encoding a model configuration.
      custom_objects: Optional dictionary mapping names
          (strings) to custom classes or functions to be
          considered during deserialization.
    Returns:
      A Keras model instance (uncompiled).
    """
    config = json.loads(json_string)
    from .layers import deserialize  # noqa
    return deserialize(config, custom_objects=custom_objects)
