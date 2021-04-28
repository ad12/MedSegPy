import inspect
from typing import Callable, Sequence, Union

import tensorflow as tf

from medsegpy.utils import env

try:
    import tf.keras.activations as activations
    import tf.keras.backend as K
except ImportError:
    import keras.activations as activations
    import keras.backend as K


def to_numpy(x):
    """Extract numpy equivalent from tensor. Currently does not support TF1.

    Note:
        This method is to simplify data extraction from fixed tensors.
        It should not be used for tensor computations.

    Args:
        x (Tensor): The tensor.

    Returns:
        ndarray: The numpy array.
    """
    if env.is_tf2() and tf.executing_eagerly():
        return x.numpy()
    return K.eval(x)


def get_shape(x):
    """Returns shape of Keras tensor."""
    if hasattr(K, "int_shape"):
        return K.int_shape(x)
    else:
        return K.get_variable_shape(x)


def reduce_tensor(x, reduction="mean", axis=None, weights=None):
    """Reduce tensor along axis.

    If specified, the `weights` tensor will be broadcast when multipied with `x`.
    Weighted mean is computed over `axis` (if specified), else, it is
    computed over the full tensor.

    Args:
        x (Tensor): The tensor.
        reduction (str, optional): Reduction to apply to the tensor.
            One of `"none"` | `"mean"` | `"sum"`.
            * `"none"`: no reduction applied
            * `"mean"`: The weighted mean of the output is taken
            * `"sum"`: The weighted output will be summed
        axis (int(s), optional): The axis to apply reduction to.
        weights (Tensor, optional): A weights tensor broadcastable to shape of x.

    Returns:
        Tensor: Reduced tensor.

    Raises:
        ValueError: If `reduction` not one of `"none"` | `"mean"` | `"sum"`.
    """
    use_weights = weights is not None
    if use_weights:
        x *= weights

    if reduction == "mean" and use_weights:
        ndim = K.ndim(x)
        if axis is None:
            axis = tuple(range(-ndim, 0))
        else:
            axis = _to_negative_index(axis)
        weights_shape = get_shape(weights)
        valid_dims = tuple(
            dim
            for dim in axis
            if abs(dim) <= len(weights_shape) and weights_shape[dim] not in (1, None)
        )
        broadcast_dims = tuple(dim for dim in axis if dim not in valid_dims)
        reduced = K.sum(x, axis=valid_dims, keepdims=True) / K.sum(weights, axis=valid_dims)
        if broadcast_dims:
            reduced = K.mean(reduced, axis=broadcast_dims)
        return reduced
    elif reduction == "mean":
        return K.mean(x, axis=axis)
    elif reduction == "sum":
        return K.sum(x, axis=axis)
    elif reduction in ("none", None):
        return x
    raise ValueError(f"Reduction '{reduction}' unknown.")


def get_activation(act: Union[str, Callable]):
    if isinstance(act, Callable):
        act_name = act.__name__
        act_fn = act
    else:
        act_name = act
        act_fn = activations.__dict__[act]

    args = inspect.signature(act_fn).parameters.keys()
    return act_name, act_fn, args


def _to_negative_index(axis, ndim):
    """Convert positive axes to their negative counterparts."""
    if not isinstance(axis, Sequence):
        axis = (axis,)
    axis = tuple({dim if dim < 0 else -ndim + dim for dim in axis})
    if len(axis) == 1:
        axis = axis[0]
    return axis
