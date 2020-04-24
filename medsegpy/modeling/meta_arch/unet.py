from typing import Union, Sequence, Dict

import numpy as np
import tensorflow as tf
from keras import Input
from keras.engine import get_source_inputs
from keras.layers import Concatenate, Conv2DTranspose, MaxPooling2D, Conv2D, \
    Activation, Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization as BN, \
    Dropout

from medsegpy.config import UNetConfig, UNet3DConfig
from ..model import add_sem_seg_activation, zero_pad_like, Model
from .build import META_ARCH_REGISTRY


def build_encoder_block(
    x: tf.Tensor,
    num_filters: Union[int, Sequence[int]],
    kernel_size: Union[int, Sequence[int]] = 3,
    num_conv: int = 2,
    activation: str = "relu",
    kernel_initializer: Union[str, Dict] = "he_normal",
    dropout: float = 0.0,
):
    """Builds simple FCN encoder block.

    Structure is below. Where blocks in `[]` are repeated:

    [Conv -> Activation] -> BN -> Dropout.

    Args:
        x (tf.Tensor): Input tensor.
        num_filters (int or Sequence[int]): Number of filters to use
            for each conv layer. If a sequence, will override number of conv
            layers specified by `num_conv`.
        kernel_size: Kernel size accepted by Keras convolution layers.
        num_conv (int, optional): Number of convolutional blocks
            (conv + activation) to use.
        activation (str, optional): Activation type.
        kernel_initializer: Kernel initializer accepted by
            `keras.layers.Conv(...)`.
        dropout (float, optional): Dropout rate.

    Returns:
        tf.Tensor: Encoder block output.
    """
    if len(x.shape) == 4:  # 2D
        conv_type = Conv2D
    elif len(x.shape) == 5:  # 3D
        conv_type = Conv3D
    else:
        raise ValueError("Only 2D or 3D inputs are supported")

    if isinstance(num_filters, int):
        num_filters = [num_filters] * num_conv

    for filters in num_filters:
        x = conv_type(
            filters,
            kernel_size,
            padding="same",
            activation=activation,
            kernel_initializer=kernel_initializer,
        )(x)
    x = BN(axis=-1, momentum=0.95, epsilon=0.001)(x)
    x = Dropout(rate=dropout)(x)
    return x


def build_decoder_block(
    x,
    x_skip,
    num_filters: Union[int, Sequence[int]],
    unpool_size: Sequence[int],
    kernel_size: Union[int, Sequence[int]] = 3,
    num_conv: int = 2,
    activation: str = "relu",
    kernel_initializer: Union[str, Dict] = "he_normal",
    dropout: float = 0.0,
):
    if len(x.shape) == 4:  # 2D
        conv_transpose_type = Conv2DTranspose
    elif len(x.shape) == 5:  # 3D
        conv_transpose_type = Conv3DTranspose
    else:
        raise ValueError("Only 2D or 3D inputs are supported")

    if isinstance(num_filters, int):
        # conv_transpose + num_conv
        num_filters = [num_filters] * (num_conv + 1)

    x_shape = x.shape.as_list()[1:-1]
    x = conv_transpose_type(
        num_filters[0],
        kernel_size,
        padding="same",
        strides=unpool_size,
        kernel_initializer=kernel_initializer,
    )(x)
    x_shape = [x * y for x, y in zip(x_shape, unpool_size)]
    x = Concatenate(axis=-1)([zero_pad_like(x, x_skip, x_shape), x_skip])

    if len(num_filters) > 1:
        x = build_encoder_block(
            x,
            num_filters[1:],
            kernel_size,
            num_conv,
            activation,
            kernel_initializer,
            dropout,
        )
    return x


@META_ARCH_REGISTRY.register()
class UNet2D(object):
    def __init__(self, cfg: UNetConfig):
        self._cfg = cfg
        self._pooler_type = MaxPooling2D
        self._conv_type = Conv2D

        self._dim = 2
        self.kernel_size = (3, 3)

    def build_model(self, input_tensor=None) -> Model:
        cfg = self._cfg
        seed = cfg.SEED
        input_size = cfg.IMG_SIZE
        depth = cfg.DEPTH
        kernel_size = self.kernel_size

        kernel_initializer = {
            "class_name": cfg.KERNEL_INITIALIZER,
            "config": {"seed": seed}
        }
        output_mode = cfg.LOSS[1]
        assert output_mode in ["sigmoid", "softmax"]
        num_classes = cfg.get_num_classes()
        num_filters = cfg.NUM_FILTERS
        if not num_filters:
            num_filters = [2 ** feat * 32 for feat in range(depth)]
        else:
            depth = len(num_filters)

        # Build inputs.
        if not input_tensor:
            inputs = self._build_input(cfg.IMG_SIZE)
        else:
            inputs = input_tensor

        # Encoder.
        x_skips = []
        pool_sizes = []
        x = inputs
        for depth_cnt in range(depth):
            x = build_encoder_block(
                x,
                num_filters[depth_cnt],
                kernel_size=kernel_size,
                num_conv=2,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=0.0,
            )

            # Maxpool until penultimate depth.
            if depth_cnt < depth - 1:
                x_skips.append(x)
                pool_size = self._get_pool_size(x)
                pool_sizes.append(pool_size)
                x = self._pooler_type(pool_size=pool_size)(x)

        # Decoder.
        for i, (x_skip, unpool_size) in enumerate(
            zip(x_skips[::-1], pool_sizes[::-1])
        ):
            depth_cnt = depth - i - 2
            x = build_decoder_block(
                x,
                x_skip,
                num_filters[depth_cnt],
                unpool_size,
                kernel_size=kernel_size,
                num_conv=2,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=0.0,
            )

        # 1x1 convolution to get pixel-wise semantic segmentation.
        x = add_sem_seg_activation(
            x,
            num_classes,
            conv_type=self._conv_type,
            activation=output_mode,
            kernel_initializer="glorot_uniform",
            seed=seed,
        )

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        model = Model(inputs=inputs, outputs=[x])

        return model

    def _build_input(self, input_size):
        if len(input_size) == 2:
            input_size = input_size + (1,)
        elif len(input_size) != 3:
            raise ValueError("2D U-Net must have an input of shape HxWxC")

        return Input(input_size)

    def _get_pool_size(self, x: tf.Tensor):
        return [
            2 if d % 2 == 0 or d % 3 != 0 else 3
            for d in x.shape.as_list()[1:-1]
        ]


@META_ARCH_REGISTRY.register()
class UNet3D(UNet2D):
    def __init__(self, cfg: UNet3DConfig):
        super().__init__(cfg)
        self._cfg = cfg
        self._pooler_type = MaxPooling3D
        self._conv_type = Conv3D

        self._dim = 3
        self.kernel_size = (3, 3, 3)

    def _build_input(self, input_size):
        if len(input_size) == 3:
            input_size = input_size + (1,)
        elif len(input_size) != 4:
            raise ValueError("2D U-Net must have an input of shape HxWxDxC")

        return Input(input_size)

    def _get_pool_size(self, x: tf.Tensor):
        return (2, 2, 1) if x.shape.as_list()[-2] // 2 == 0 else (2, 2, 2)
