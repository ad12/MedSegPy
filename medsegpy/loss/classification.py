import logging
from typing import Callable, Union

import numpy as np
import tensorflow as tf

from medsegpy.utils import env
from medsegpy.loss.utils import get_activation, get_shape, reduce_tensor

try:
    import tf.keras.backend as K
except ImportError:
    import keras.backend as K

logger = logging.getLogger(__name__)


class DiceLoss():
    def __init__(
        self,
        weights=None,
        activation: str = None,
        flatten: Union[str, bool] = False,
        remove_background: bool = False,
        reduction: str = "mean",
        reduction_axis=None,
        eps: float = None,
    ):
        """
        Args:
            weights (array-like, optional): Class weighting to use.
            activation (bool, optional): Activation function to apply.
                One of `"sigmoid"` | `"softmax"` | `"none"`. If "none",
                no activation will be applied to input.
            flatten (`bool` or `str`, optional): Dimensions to collapse
                into the spatial dimension prior to dice computation.
                One of `True` | `"batch"` | `"channel"`.
                If `True`, flatten both batch and channel dimensions.
            remove_background (bool, optional): If `True`, drop the first channel
                from both y_true and y_pred tensors.
            reduction (str, optional): Reduction to apply to the loss.
            One of `"none"` | `"mean"` | `"sum"`.
                * `"none"`: no reduction applied
                * `"mean"`: The weighted mean of the output is taken
                * `"sum"`: The output will be summed
            reduction_axis (int, optional): The axis to apply reduction to.
            eps (float, optional): Smoothing factor. Defaults to `keras.backend.epsilon()`.
        """
        self.weights = np.asarray(weights) if weights is not None else weights
        self.remove_background = remove_background
        self.eps = K.epsilon() if eps is None else eps

        if activation not in ("none", None):
            self.activation, self._act_fn, act_args = get_activation(activation)
        else:
            self.activation, self._act_fn, act_args = "none", None, ()
        self._act_kwargs = {}
        if "axis" in act_args:
            # Activation computed over channel dimension
            self._act_kwargs["axis"] = -1

        assert isinstance(flatten, bool) or flatten in ("batch", "channel")
        if weights is not None and flatten == "channel":
            raise ValueError(
                "Cannot flatten across 'channel' dimension and weight channels"
            )
        self.flatten = flatten

        assert reduction in (None, "none", "mean", "sum")
        self.reduction = reduction
        self.reduction_axis = reduction_axis
    
    def __call__(self, y_true, y_pred):
        if self._act_fn is not None:
            y_pred = self._act_fn(y_pred, **self._act_kwargs)

        if self.remove_background:
            y_true, y_pred = y_true[..., 1:], y_pred[..., 1:]
        if env.is_tf2():
            y_true = tf.dtypes.cast(y_true, y_pred.dtype)

        spatial_dims = tuple(range(1, K.ndim(y_pred) - 1))  # spatial dims
        if self.flatten == "batch": dims = (0,) + spatial_dims
        elif self.flatten == "channel": dims = spatial_dims + (K.ndim(y_pred) - 1,)
        elif self.flatten: dims = (0,) + spatial_dims + (K.ndim(y_pred) - 1,)
        else: dims = spatial_dims

        intersection = K.sum(y_true * y_pred, axis=dims)
        overlap = K.sum(y_true, axis=dims) + K.sum(y_pred, axis=dims)
        dice = (2.0 * intersection + self.eps) / (overlap + self.eps)
        loss = 1 - dice  # largest shape: (N, C)

        weights = K.variable(self.weights) if self.weights is not None else None
        return reduce_tensor(
            loss,
            reduction=self.reduction,
            axis=self.reduction_axis,
            weights=weights,
        )
