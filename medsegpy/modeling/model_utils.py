from typing import Dict

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Conv3D, ZeroPadding2D, ZeroPadding3D


def build_keras_config(class_name, **kwargs) -> Dict:
    """Builds config dictionary compatible with keras layer arguments."""
    return {"class_name": class_name, "config": kwargs}


def get_primary_shape(x: tf.Tensor):
    """Get sizes of the primary dimensions of x.

    Primary dimensions of a tensor are all dimensions that do not correspond to
    the batch dimension `B` or the channel dimension `C`.

    Args:
        x (tf.Tensor): Shape Bx(...)xC (channels_last) or BxCx(...) (channels_first).  # noqa

    Returns:
        list: primary dimensions.
    """
    x_shape = x.shape.as_list()
    x_shape = (
        x_shape[1:-1]
        if K.image_data_format() == "channels_last"
        else x_shape[2:]
    )

    return x_shape


def zero_pad_like(x: tf.Tensor, y: tf.Tensor, x_shape=None, y_shape=None):
    """Zero pads input (x) to size of target (y).

    Padding is symmetric when difference in dimension size is multiple of 2.
    Otherwise, the bottom padding is 1 larger than the top padding.
    Assumes channels are last dimension.

    Primary dimensions of a tensor are all dimensions that do not correspond to
    the batch dimension or the channel dimension.

    Args:
        x (tf.Tensor): Input tensor.
        y (tf.Tensor): Target tensor.
        x_shape (Sequence[int]): Expected shape of `x`. Required when primary
            `x` dimensions have sizes `None`.
        y_shape (Sequence[int]): Like `x_shape`, but for `y`.

    Returns:
        tf.Tensor: Zero-padded tensor
    """
    if not x_shape:
        x_shape = get_primary_shape(x)
    if not y_shape:
        y_shape = get_primary_shape(y)

    assert not any(s is None for s in x_shape)
    assert not any(s is None for s in y_shape)

    if x_shape == y_shape:
        return x
    diff = [y_s - x_s for x_s, y_s in zip(x_shape, y_shape)]
    assert all(
        d >= 0 for d in diff
    ), "x must be smaller than y in all dimensions"

    if len(diff) == 2:
        padder = ZeroPadding2D
    elif len(diff) == 3:
        padder = ZeroPadding3D
    else:
        raise ValueError("Zero padding available for 2D or 3D images only")

    padding = [d // 2 if d % 2 == 0 else (d // 2, d // 2 + 1) for d in diff]
    x = padder(padding)(x)
    return x


def add_sem_seg_activation(
    x: tf.Tensor,
    num_classes: int,
    activation: str = "sigmoid",
    conv_type=None,
    kernel_initializer=None,
    seed=None,
) -> tf.Tensor:
    """Standardized output layer for semantic segmentation using 1x1 conv.

    Args:
        x (tf.Tensor): Input tensor.
        num_classes (int): Number of classes
        activation (str, optional): Activation type. Typically `'sigmoid'` or
            `'softmax'`.
        conv_type: Either `Conv2D` or `Conv3D`.
        kernel_initializer: Kernel initializer accepted by
            `Conv2D` or `Conv3D`.
        seed (int, optional): Kernel intialization seed. Ignored if
            `kernel_initializer` is a config dict.
    """

    # Initializing kernel weights to 1 and bias to 0.
    # i.e. without training, the x would be a sigmoid activation on each
    # pixel of the input
    if not conv_type:
        conv_type = Conv2D
    else:
        assert conv_type in [Conv2D, Conv3D]
    if not kernel_initializer:
        kernel_initializer = {
            "class_name": "glorot_uniform",
            "config": {"seed": seed},
        }
    elif isinstance(kernel_initializer, str):
        kernel_initializer = {
            "class_name": kernel_initializer,
            "config": {"seed": seed},
        }

    x = conv_type(
        num_classes,
        1,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="output_activation",
    )(x)
    return x
