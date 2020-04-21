import logging

import numpy as np
from keras.engine.topology import get_source_inputs
from keras.initializers import he_normal
from keras.layers import BatchNormalization as BN
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Concatenate
from keras.models import Model
from keras.utils import plot_model

logger = logging.getLogger(__name__)


def anisotropic_unet_2d(
    input_size = None,
    input_tensor = None,
    output_mode = None,
    num_filters = None,
    depth = 6,
    kernel_size = (3, 3),
    pooling_size = None,
    pooling_ratio = None,
):
    """Generate Unet 2D model compatible with Keras 2

    :param input_size: tuple of input size - format: (height, width, 1)

    :rtype: Keras model

    :raise ValueError if input_size is not tuple or dimensions of input_size do not match (height, width, 1)
    """
    from medsegpy import glob_constants
    logger.info('Initializing unet with seed: %s' % str(glob_constants.SEED))
    SEED = glob_constants.SEED
    if input_tensor is None and (type(input_size) is not tuple or len(input_size) != 3):
        raise ValueError('input_size must be a tuple of size (height, width, 1)')

    if num_filters is None:
        nfeatures = [2 ** feat * 32 for feat in np.arange(depth)]
    else:
        nfeatures = num_filters
        assert len(nfeatures) == depth

    conv_ptr = []

    # input layer
    inputs = input_tensor if input_tensor is not None else Input(input_size)

    # step down convolutional layers
    unpooling_sizes = []
    pool = inputs
    for depth_cnt in range(depth):

        conv = Conv2D(nfeatures[depth_cnt], kernel_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(pool)
        conv = Conv2D(nfeatures[depth_cnt], kernel_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.0)(conv)

        conv_ptr.append(conv)

        # Only maxpool till penultimate depth
        if depth_cnt < depth - 1:
            x_shape = conv.shape.as_list()
            pool_size = pooling_size if pooling_size else __get_pooling_size__(x_shape, pooling_ratio)
            unpooling_sizes.append(pool_size)
            pool = MaxPooling2D(pool_size=pool_size, padding='same')(conv)

    # step up convolutional layers
    for depth_cnt in range(depth - 2, -1, -1):
        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None
        unpooling_size = unpooling_sizes[depth_cnt]
        up = Concatenate(axis=3)([Conv2DTranspose(nfeatures[depth_cnt], kernel_size,
                                                  padding='same',
                                                  strides=unpooling_size,
                                                  kernel_initializer=he_normal(seed=SEED))(conv),
                                  conv_ptr[depth_cnt]])

        conv = Conv2D(nfeatures[depth_cnt], kernel_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(up)
        conv = Conv2D(nfeatures[depth_cnt], kernel_size,
                      padding='same',
                      activation='relu',
                      kernel_initializer=he_normal(seed=SEED))(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.00)(conv)

    # combine features
    # this if statement is required for legacy purposes
    # some weights were trained with a joint activation, which makes it difficult to load weights effectively
    if output_mode is not None:
        recon = Conv2D(1, (1, 1), padding='same', activation=output_mode, kernel_initializer=he_normal(seed=SEED))(conv)
    else:
        recon = conv

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs=inputs, outputs=[recon])

    return model


def __get_pooling_size__(x_shape: list, pool_ratio=None):
    yres = x_shape[1]
    xres = x_shape[2]

    pooling_size = []
    for r in yres, xres:
        if r % 2 == 0:
            pooling_size.append(2)
        else:
            pooling_size.append(3)

    pooling_size = tuple(pooling_size)

    if not pool_ratio:
        return pooling_size

    if yres > xres:
        return pooling_size[0] * pool_ratio, pooling_size[1]
    else:
        return pooling_size[0], pooling_size[1] * pool_ratio


if __name__ == '__main__':
    save_path = '../imgs/anisotropic_unet2d.png'
    m = anisotropic_unet_2d(input_size=(72, 288, 1), kernel_size=(3, 3))
    plot_model(m, to_file=save_path, show_shapes=True)

