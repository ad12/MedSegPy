import logging
from abc import ABC, abstractmethod

from keras.initializers import glorot_uniform, he_normal
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization

from ..layers import MaxPoolingWithArgmax2D, MaxUnpooling2D
from ..model import Model

logger = logging.getLogger(__name__)


class KModel(ABC):
    def __init__(self, input_shape=None, input_tensor=None, seed=None):
        self._input_shape = input_shape
        self._input_tensor = input_tensor
        self._seed = seed

    @abstractmethod
    def build_model(self):
        pass


class SegNet(KModel):
    def __init__(self, input_shape=None, input_tensor=None, seed=None, **kwargs):
        super().__init__(input_shape, input_tensor, seed)

        self._n_labels = kwargs.get("n_labels")
        self._depth = kwargs.get("depth")
        self._num_conv_layers = kwargs.get("num_conv_layers")
        self._num_filters = kwargs.get("num_filters")
        self._kernel = kwargs.get("kernel", 3)
        self._pool_size = kwargs.get("pool_size", (2, 2))
        self._output_mode = kwargs.get("output_mode")
        self._single_bn = kwargs.get("single_bn", False)
        self._conv_act_bn = kwargs.get("conv_act_bn", False)

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
            if conv_act_bn:
                curr_layer, l_mask = self._encoder_block_conv_act_bn(
                    curr_layer,
                    level=i + 1,
                    num_conv_layers=num_conv_layers[i],
                    num_filters=num_filters[i],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                )
            else:
                curr_layer, l_mask = self._encoder_block(
                    curr_layer,
                    level=i + 1,
                    num_conv_layers=num_conv_layers[i],
                    num_filters=num_filters[i],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                    single_bn=single_bn,
                )

            mask_layers.append(l_mask)

        logger.info("Building decoder...")
        # decoder
        for i in reversed(range(depth)):
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
                    num_filters_next=num_filters[i] if i == 0 else num_filters[i - 1],
                    kernel=kernel,
                    pool_size=eff_pool_size,
                    single_bn=single_bn,
                )

        outputs = Convolution2D(
            n_labels, (1, 1), kernel_initializer=glorot_uniform(seed=seed), activation=output_mode
        )(curr_layer)

        segnet = Model(inputs=inputs, outputs=outputs, name="segnet")

        return segnet

    def _encoder_block(
        self,
        x,
        level,
        num_conv_layers=2,
        num_filters=64,
        kernel=3,
        pool_size=(2, 2),
        single_bn=False,
        add_pool: bool = True,
    ):
        seed = self._seed
        if num_conv_layers <= 0:
            raise ValueError("Must have at least 1 conv layer")

        curr_layer = x

        for i in range(num_conv_layers):
            conv = Convolution2D(
                num_filters,
                (kernel, kernel),
                padding="same",
                kernel_initializer=he_normal(seed=seed),
                name="enc_%d_conv_%d" % (level, i + 1),
            )(curr_layer)
            if not single_bn:
                conv = BatchNormalization(name="enc_%d_bn_%d" % (level, i + 1))(conv)
            conv = Activation("relu", name="enc_%d_relu_%d" % (level, i + 1))(conv)
            curr_layer = conv

        if single_bn:
            curr_layer = BatchNormalization(name="enc_%d_bn" % level)(curr_layer)

        if add_pool:
            l_pool, l_mask = MaxPoolingWithArgmax2D(
                pool_size=pool_size, strides=pool_size, name="enc_%d_pool" % level
            )(curr_layer)
        else:
            l_pool, l_mask = curr_layer, None

        return l_pool, l_mask

    def _decoder_block(
        self,
        x_pool,
        x_mask,
        level,
        num_conv_layers=2,
        num_filters=64,
        num_filters_next=32,
        kernel=3,
        pool_size=(2, 2),
        single_bn=False,
    ):
        seed = self._seed
        unpool_1 = MaxUnpooling2D(pool_size)([x_pool, x_mask])

        curr_layer = unpool_1

        for i in range(num_conv_layers):
            used_num_filters = num_filters_next if i == num_conv_layers - 1 else num_filters
            conv = Convolution2D(
                used_num_filters,
                (kernel, kernel),
                kernel_initializer=he_normal(seed=seed),
                padding="same",
                name="dec_%d_conv_%d" % (level, i + 1),
            )(curr_layer)

            if not single_bn:
                conv = BatchNormalization(name="dec_%d_bn_%d" % (level, i + 1))(conv)

            conv = Activation("relu", name="dec_%d_relu_%d" % (level, i + 1))(conv)
            curr_layer = conv

        if single_bn:
            curr_layer = BatchNormalization(name="dec_%d_bn" % level)(curr_layer)

        return curr_layer

    def _encoder_block_conv_act_bn(
        self,
        x,
        level,
        num_conv_layers=2,
        num_filters=64,
        kernel=3,
        pool_size=(2, 2),
        add_pool: bool = True,
    ):
        seed = self._seed

        if num_conv_layers <= 0:
            raise ValueError("Must have at least 1 conv layer")

        curr_layer = x

        for i in range(num_conv_layers):
            conv = Convolution2D(
                num_filters,
                (kernel, kernel),
                padding="same",
                kernel_initializer=he_normal(seed=seed),
                name="enc_%d_conv_%d" % (level, i + 1),
            )(curr_layer)
            conv = Activation("relu", name="enc_%d_relu_%d" % (level, i + 1))(conv)
            conv = BatchNormalization(name="enc_%d_bn_%d" % (level, i + 1))(conv)
            curr_layer = conv

        if add_pool:
            l_pool, l_mask = MaxPoolingWithArgmax2D(
                pool_size=pool_size, strides=pool_size, name="enc_%d_pool" % level
            )(curr_layer)
        else:
            l_pool, l_mask = curr_layer, None

        return l_pool, l_mask

    def _decoder_block_conv_act_bn(
        self,
        x_pool,
        x_mask,
        level,
        num_conv_layers=2,
        num_filters=64,
        num_filters_next=32,
        kernel=3,
        pool_size=(2, 2),
    ):
        seed = self._seed

        unpool_1 = MaxUnpooling2D(pool_size)([x_pool, x_mask])

        curr_layer = unpool_1

        for i in range(num_conv_layers):
            used_num_filters = num_filters_next if i == num_conv_layers - 1 else num_filters
            conv = Convolution2D(
                used_num_filters,
                (kernel, kernel),
                padding="same",
                kernel_initializer=he_normal(seed=seed),
                name="dec_%d_conv_%d" % (level, i + 1),
            )(curr_layer)
            conv = Activation("relu", name="dec_%d_relu_%d" % (level, i + 1))(conv)
            conv = BatchNormalization(name="dec_%d_bn_%d" % (level, i + 1))(conv)
            curr_layer = conv

        return curr_layer


def Segnet_v2(**kwargs):
    m = SegNet(**kwargs)
    return m.build_model()
