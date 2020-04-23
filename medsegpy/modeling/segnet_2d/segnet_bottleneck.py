import logging

from keras.initializers import glorot_uniform
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.models import Model

from .segnet import SegNet

logger = logging.getLogger(__name__)


class SegNetBottleneck(SegNet):
    """SegNet with single bottleneck (i.e. no pooling at final depth)."""

    def build_model(self):
        input_tensor = self._input_tensor
        input_shape = self._input_shape
        seed = self._seed
        n_labels = self._n_labels
        depth = self._depth
        num_conv_layers = self._num_conv_layers
        num_filters = self._num_filters
        kernel = self._kernel
        pool_size = self._pool_size
        output_mode = self._output_mode
        single_bn = self._single_bn
        conv_act_bn = self._conv_act_bn

        logger.info("Initializing SegNet with seed: {}".format(self._seed))

        inputs = input_tensor if input_tensor else Input(shape=input_shape)

        mask_layers = []

        curr_layer = inputs
        eff_pool_sizes = []

        # Determine pool sizes
        for i in range(depth):
            level = i + 1
            eff_pool_size = pool_size
            divisor = pool_size[0] ** level
            if input_shape[0] % divisor != 0:
                eff_pool_size = (3, 3)
            eff_pool_sizes.append(eff_pool_size)

        # encoder
        logger.info("Building Encoder...")
        for i in range(depth):
            eff_pool_size = eff_pool_sizes[i]
            # Do not pool on bottleneck layer.
            is_bottleneck = i == depth - 1
            n_conv = (
                num_conv_layers[i] - 1 if is_bottleneck else num_conv_layers[i]
            )
            if conv_act_bn:
                curr_layer, l_mask = self._encoder_block_conv_act_bn(
                    curr_layer,
                    level=i + 1,
                    num_conv_layers=n_conv,
                    num_filters=num_filters[i],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                    add_pool=not is_bottleneck,
                )
            else:
                curr_layer, l_mask = self._encoder_block(
                    curr_layer,
                    level=i + 1,
                    num_conv_layers=n_conv,
                    num_filters=num_filters[i],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                    single_bn=single_bn,
                    add_pool=not is_bottleneck,
                )

            mask_layers.append(l_mask)

        # Add additional layer at end:
        if conv_act_bn:
            curr_layer, _ = self._encoder_block_conv_act_bn(
                curr_layer,
                level=depth + 1,
                num_conv_layers=1,
                num_filters=num_filters[depth - 2],
                kernel=kernel,
                pool_size=eff_pool_size,
                add_pool=False,
            )
        else:
            curr_layer, _ = self._encoder_block(
                curr_layer,
                level=depth + 1,
                num_conv_layers=1,
                num_filters=num_filters[depth - 2],
                kernel=kernel,
                pool_size=eff_pool_size,
                single_bn=single_bn,
                add_pool=False,
            )

        logger.info("Building decoder...")
        # decoder
        for i in reversed(range(depth - 1)):
            l_mask = mask_layers[i]
            eff_pool_size = eff_pool_sizes[i]
            if conv_act_bn:
                curr_layer = self._decoder_block_conv_act_bn(
                    curr_layer,
                    l_mask,
                    level=i + 1,
                    num_conv_layers=num_conv_layers[i],
                    num_filters=num_filters[i],
                    num_filters_next=1 if i == 0 else num_filters[i - 1],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                )
            else:
                curr_layer = self._decoder_block(
                    curr_layer,
                    l_mask,
                    level=i + 1,
                    num_conv_layers=num_conv_layers[i],
                    num_filters=num_filters[i],
                    num_filters_next=num_filters[i]
                    if i == 0
                    else num_filters[i - 1],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                    single_bn=single_bn,
                )

        outputs = Convolution2D(
            n_labels,
            (1, 1),
            kernel_initializer=glorot_uniform(seed=seed),
            activation=output_mode,
        )(curr_layer)

        model = Model(inputs=inputs, outputs=outputs, name="segnet_bottleneck")

        return model
