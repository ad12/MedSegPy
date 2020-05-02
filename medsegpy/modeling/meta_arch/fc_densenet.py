"""Implementation of Fully-Convolutional Densenet.

Built from paper: https://arxiv.org/pdf/1611.09326.pdf
"""
from keras.layers import (
    Activation,
    BatchNormalization as BN,
    Conv2D,
    Dropout,
    Concatenate,
    MaxPooling2D,
    MaxPooling3D,
    Conv3D,
    Conv2DTranspose,
    Conv3DTranspose,
    Input,
)
from keras.engine import get_source_inputs
from keras import backend as K
import tensorflow as tf
from typing import Dict, Sequence, Union

from medsegpy.config import FCDenseNetConfig
from .build import ModelBuilder, META_ARCH_REGISTRY
from ..model import Model
from ..model_utils import add_sem_seg_activation


def build_fc_dense_block(
    x: tf.Tensor,
    filters: Union[int, Sequence[int]],
    kernel_size: Union[int, Sequence[int]] = 3,
    num_layers: int = 2,
    activation: str = "relu",
    kernel_initializer: Union[str, Dict] = "he_normal",
    dropout: float = 0.0,
    use_bn: bool = True,
    concat_input: bool = False,
    conv_type=Conv2D,
):
    """Builds fully convolutional dense block.

    The sequence is shown below.
    The blocks are repeated `` number of times.
    Layers in braces (`{}`) are optional.`(C)` refers to
    concatenation:

           -------------------------------------------  ---------
           |                                         |  |       |
           |                                         v  |       v
        x  -> {BN} -> {Act} -> Conv -> {Dropout} -> (C) -> ... (C) ... (C) ->
           |                                     |              |       ^
           |                                     |              |       |
           |                                     |              ---------
           |                                     |                      |
           |                                     |                      |
           |                                     ------------------------
           |                                                            |
           |                                                            |
           --------------------------------------------------------------

    Args:
        x (tf.Tensor): Input tensor.
        filters (int or Sequence[int]): Number of filters to use
            for each conv layer. If a sequence, will override number of conv
            layers specified by `num_layers`.
        kernel_size: Kernel size accepted by Keras convolution layers.
        num_layers (int, optional): Number of layers/subblocks to make.
        activation (str, optional): Activation type. If `""`, no activation is
            used.
        kernel_initializer: Kernel initializer accepted by
            `keras.layers.Conv(...)`.
        dropout (float, optional): Dropout rate.
        use_bn (bool, optional): If `True`, use batch normalization.
        concat_input (bool, optional): Concatenate input into dense block
            with final layer.
    """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    num_layers = len(filters) if isinstance(filters, Sequence) else num_layers
    assert isinstance(num_layers, int), "{}".format(type(num_layers))
    if isinstance(filters, int):
        filters = [filters] * num_layers

    stack = x
    layer_outputs = [x] if concat_input else []

    for i in range(num_layers):
        if use_bn:
            x = BN(axis=channel_axis, momentum=0.95, epsilon=0.001)(x)
        if activation:
            x = Activation(activation)(x)
        x = conv_type(
            filters[i],
            kernel_size,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)
        if dropout != 0:
            x = Dropout(rate=dropout)(x)

        layer_outputs.append(x)
        x = stack = Concatenate(axis=channel_axis)([stack, x])

    if len(layer_outputs) > 1:
        x = Concatenate(axis=channel_axis)(layer_outputs)
    else:
        x = layer_outputs[0]
    return x


@META_ARCH_REGISTRY.register()
class FCDenseNet(ModelBuilder):
    def __init__(self, cfg):
        super().__init__(cfg)
        if len(cfg.IMG_SIZE) == 3:
            self._pooler_type = MaxPooling2D
            self._conv_type = Conv2D
            self._conv_transpose_type = Conv2DTranspose
            self._dim = 2
        elif len(cfg.IMG_SIZE) == 4:
            self._pooler_type = MaxPooling3D
            self._conv_type = Conv3D
            self._conv_transpose_type = Conv3DTranspose
            self._dim = 3
        else:
            raise ValueError(
                "Image size {} not supported. "
                "Must be 2D (HxWxC) or 3D (HxWxDxC)"
            )

        self.kernel_size = (3, 3)

    def _get_enc_dec_counts(
        self,
        counts: Union[Sequence[int], Sequence[Sequence[int]]],
        depth: int,
        cfg_field: str
    ):
        """Returns the count for the encoder and decoder.

        Encoder count includes the bottleneck.
        """
        if len(counts) == 1:
            num_enc_counts = [counts[0]] * depth
            num_dec_counts = [counts[0]] * (depth - 1)
        elif len(counts) == depth:
            num_enc_counts = counts
            num_dec_counts = counts[-2::-1]
        elif len(counts) == 2*depth-1:
            num_enc_counts = counts[:depth]
            num_dec_counts = counts[depth:]
        else:
            raise ValueError(
                "len(cfg.{}) should be 1, cfg.DEPTH, or 2*cfg.DEPTH-1. "
                "cfg.DEPTH={}, len(cfg.NUM_FILTERS)={}".format(
                    cfg_field, depth, len(counts)
                )
            )
        return num_enc_counts, num_dec_counts

    def build_model(self, input_tensor=None):
        cfg: FCDenseNetConfig = self._cfg
        seed = cfg.SEED
        depth = cfg.DEPTH
        kernel_size = self.kernel_size
        num_filters = cfg.NUM_FILTERS
        num_layers = cfg.NUM_LAYERS
        dropout = cfg.DROPOUT
        enc_num_filters, dec_num_filters = self._get_enc_dec_counts(
            num_filters, depth, "NUM_FILTERS",
        )
        enc_num_layers, dec_num_layers = self._get_enc_dec_counts(
            num_layers, depth, "NUM_LAYERS",
        )
        kernel_initializer = {
            "class_name": cfg.KERNEL_INITIALIZER,
            "config": {"seed": seed},
        }
        output_mode = cfg.LOSS[1]
        assert output_mode in ["sigmoid", "softmax"]
        num_classes = cfg.get_num_classes()

        conv_type = self._conv_type
        pool_type = self._pooler_type
        ct_type = self._conv_transpose_type
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1

        # Build inputs.
        if not input_tensor:
            inputs = self._build_input(cfg.IMG_SIZE)
        else:
            inputs = input_tensor

        x = inputs
        x = conv_type(
            cfg.NUM_FILTERS_HEAD_CONV,
            kernel_size,
            padding="same",
            kernel_initializer=kernel_initializer,
        )(x)

        # Encoder.
        skip_connections = []
        running_n_filters = [cfg.NUM_FILTERS_HEAD_CONV]
        running_pool_sizes = []
        for i in range(depth-1):
            n_filters = enc_num_filters[i]
            n_layers = enc_num_layers[i]
            x = build_fc_dense_block(
                x,
                filters=n_filters,
                kernel_size=kernel_size,
                num_layers=n_layers,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=dropout,
                use_bn=True,
                concat_input=True,
                conv_type=conv_type,
            )
            skip_connections.append(x)
            if isinstance(n_filters, Sequence):
                running_n_filters.append(sum(n_filters))
            else:
                running_n_filters.append(n_filters * n_layers)

            # Transition down.
            x = build_fc_dense_block(
                x,
                filters=sum(running_n_filters),
                kernel_size=1,
                num_layers=1,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=dropout,
                use_bn=True,
                concat_input=False,
                conv_type=conv_type,
            )
            pool_size = self._get_pool_size(x)
            running_pool_sizes.append(pool_size)
            x = pool_type(pool_size=pool_size)(x)

        # Bottleneck.
        n_filters = enc_num_filters[depth-1]
        n_layers = enc_num_layers[depth-1]
        x = build_fc_dense_block(
            x,
            filters=n_filters,
            kernel_size=kernel_size,
            num_layers=n_layers,
            activation="relu",
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            use_bn=True,
            concat_input=False,
            conv_type=self._conv_type,
        )
        if isinstance(n_filters, Sequence):
            running_n_filters.append(sum(n_filters))
        else:
            running_n_filters.append(n_filters * n_layers)

        # Decoder.
        running_n_filters = running_n_filters[:0:-1]  # remove head
        skip_connections = skip_connections[::-1]
        running_pool_sizes = running_pool_sizes[::-1]
        for i in range(depth-1):
            # Transition Up.
            x = ct_type(
                filters=running_n_filters[i],
                kernel_size=kernel_size,
                padding="same",
                strides=running_pool_sizes[i],
                kernel_initializer=kernel_initializer
            )(x)
            # TODO: add padding layer.

            x = Concatenate(axis=channel_axis)([skip_connections[i], x])

            n_filters = dec_num_filters[i]
            n_layers = dec_num_filters[i]
            x = build_fc_dense_block(
                x,
                filters=n_filters,
                kernel_size=kernel_size,
                num_layers=n_layers,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=dropout,
                use_bn=True,
                concat_input=False,
                conv_type=self._conv_type,
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

    def _get_pool_size(self, x: tf.Tensor):
        return [
            2 if d % 2 == 0 or d % 3 != 0 else 3
            for d in x.shape.as_list()[1:-1]
        ]

    def _build_input(self, input_size):
        if len(input_size) == 2:
            input_size = input_size + (1,)
        elif len(input_size) > 4:
            raise ValueError(
                "Input size must be 2D (HxW or HxWxC) or 3D (HxWxDxC)"
            )

        return Input(input_size)

