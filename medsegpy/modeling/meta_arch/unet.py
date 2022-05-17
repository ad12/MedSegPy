from typing import Dict, Sequence, Union

import numpy as np
import tensorflow as tf
from keras import Input
from keras.layers import BatchNormalization as BN
from keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Conv3D,
    Conv3DTranspose,
    Dropout,
    MaxPooling2D,
    MaxPooling3D,
)
from keras.utils.layer_utils import get_source_inputs

from medsegpy.config import UNet3DConfig, UNetConfig
from medsegpy.modeling.layers.attention import (
    CreateGatingSignal2D,
    CreateGatingSignal3D,
    DeepSupervision2D,
    DeepSupervision3D,
    MultiAttentionModule2D,
    MultiAttentionModule3D,
)
from medsegpy.modeling.model_utils import add_sem_seg_activation, zero_pad_like

from ..model import Model
from .build import META_ARCH_REGISTRY, ModelBuilder


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
            x, num_filters[1:], kernel_size, num_conv, activation, kernel_initializer, dropout
        )
    return x


@META_ARCH_REGISTRY.register()
class UNet2D(ModelBuilder):
    def __init__(self, cfg: UNetConfig):
        super().__init__(cfg)
        self._pooler_type = MaxPooling2D
        self._conv_type = Conv2D
        self._multi_attention_module = MultiAttentionModule2D
        self._create_gating_signal = CreateGatingSignal2D
        self._deep_supervision = DeepSupervision2D

        self._dim = 2
        self.kernel_size = (3, 3)
        self.use_attention = False
        self.use_deep_supervision = False

    def build_model(self, input_tensor=None) -> Model:
        cfg = self._cfg
        seed = cfg.SEED
        depth = cfg.DEPTH
        kernel_size = self.kernel_size
        self.use_attention = cfg.USE_ATTENTION
        self.use_deep_supervision = cfg.USE_DEEP_SUPERVISION

        kernel_initializer = {"class_name": cfg.KERNEL_INITIALIZER, "config": {"seed": seed}}
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
        deep_supervision_outputs = []
        scale_factors = np.ones(((depth - 1), self._dim), dtype=int)
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
                if depth_cnt < depth - 2:
                    scale_factors[depth_cnt + 1] = scale_factors[depth_cnt] * pool_size
                x = self._pooler_type(pool_size=pool_size)(x)

        # Decoder.
        for i, (x_skip, unpool_size) in enumerate(zip(x_skips[::-1], pool_sizes[::-1])):
            depth_cnt = depth - i - 2
            skip_connect = x_skip
            # The first skip connection is not passed through
            # an attention gate, as mentioned in the paper under
            # the section "Attention Gates in U-Net Model"
            if depth_cnt > 0 and self.use_attention:
                if i == 0:
                    gating_signal = self._create_gating_signal(
                        out_channels=num_filters[depth_cnt + 1]
                    )(x)
                else:
                    gating_signal = x
                attn_out, attn_coeffs = self._multi_attention_module(
                    in_channels=num_filters[depth_cnt],
                    intermediate_channels=num_filters[depth_cnt],
                    sub_sample_factor=self._get_pool_size(x_skip),
                )([x_skip, gating_signal])
                skip_connect = attn_out

            x = build_decoder_block(
                x,
                skip_connect,
                num_filters[depth_cnt],
                unpool_size,
                kernel_size=kernel_size,
                num_conv=2,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=0.0,
            )

            if self.use_deep_supervision:
                # Determine scale factor for deep supervision
                deep_supervision_outputs.append(
                    self._deep_supervision(
                        out_channels=num_classes, scale_factor=tuple(scale_factors[depth_cnt])
                    )(x)
                )

        if self.use_deep_supervision:
            x = Concatenate(axis=-1)(deep_supervision_outputs)

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
        return [2 if d % 2 == 0 or d % 3 != 0 else 3 for d in x.shape.as_list()[1:-1]]


@META_ARCH_REGISTRY.register()
class UNet3D(UNet2D):
    def __init__(self, cfg: UNet3DConfig):
        super().__init__(cfg)
        self._pooler_type = MaxPooling3D
        self._conv_type = Conv3D
        self._multi_attention_module = MultiAttentionModule3D
        self._create_gating_signal = CreateGatingSignal3D
        self._deep_supervision = DeepSupervision3D

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
