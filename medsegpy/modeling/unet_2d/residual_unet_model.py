import logging
import os

import keras.backend as K
import numpy as np
from keras.engine.topology import get_source_inputs
from keras.initializers import he_normal
from keras.layers import BatchNormalization as BN
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Concatenate, Activation, Add, \
    GlobalAveragePooling2D, Dense, Reshape, Permute, multiply
from keras.models import Model
from keras.utils import plot_model

logger = logging.getLogger(__name__)

# List of tissues that can be segmented
FEMORAL_CARTILAGE_STR = 'fc'
MENISCUS_STR = 'men'
PATELLAR_CARTILAGE_STR = 'pc'
TIBIAL_CARTILAGE_STR = 'tc'

# Absolute directory where this file lives
__ABS_DIR__ = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_DICT = {FEMORAL_CARTILAGE_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_fc_weights--0.8968.h5'),
                MENISCUS_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_men_weights--0.7692.h5'),
                PATELLAR_CARTILAGE_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_pc_weights--0.6206.h5'),
                TIBIAL_CARTILAGE_STR: os.path.join(__ABS_DIR__, 'weights/unet_2d_tc_weights--0.8625.h5')}

# Input size that is expected
# All inputs must be at least this size
DEFAULT_INPUT_SIZE = (288, 288, 1)


def squeeze_excitation_block(xi, base_name: str, seed=None, ratio=8):
    """Adapted from https://github.com/titu1994/keras-squeeze-excite-network"""
    x = xi
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = x._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D(name='%s_se_glob_avg_pool' % base_name)(x)
    se = Reshape(se_shape, name='%s_se_reshape' % base_name)(se)
    se = Dense(filters // ratio, name='%s_se_fc1' % base_name, kernel_initializer=he_normal(seed), use_bias=False)(se)
    se = Activation('relu', name='%s_se_relu' % base_name)(se)
    se = Dense(filters, use_bias=False)(se)
    se = Activation('sigmoid', name='%s_se_sigmoid' % base_name)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([x, se], name='%s_se_scale' % base_name)
    return x


def res_block(xi, base_name: str, nfeatures: int, nlayers: int = 2, kernel_size=(3, 3), seed=None,
              layer_order=['conv', 'relu', 'bn'], dropout_rate=0.0, use_squeeze_excitation=False,
              squeeze_excitation_ratio=8):
    x = xi

    layers = []
    for l in range(nlayers):
        layers_dict = {'conv': Conv2D(nfeatures, kernel_size=kernel_size, padding='same',
                                      kernel_initializer=he_normal(seed=seed),
                                      name='%s_conv_%d' % (base_name, l + 1)),
                       'bn': BN(axis=-1,
                                momentum=0.95,
                                epsilon=0.001,
                                name='%s_bn_%d' % (base_name, l + 1)),
                       'relu': Activation('relu', name='%s_relu_%d' % (base_name, l + 1)),
                       'dropout': Dropout(rate=dropout_rate, name='%s_dropout_%d' % (base_name, l + 1))}

        for layer in layer_order:
            x = layers_dict[layer](x)
            layers.append(x)

    if use_squeeze_excitation:
        x = squeeze_excitation_block(x, base_name=base_name, seed=seed, ratio=squeeze_excitation_ratio)

    x = Add()([x, xi])

    return x


def residual_unet_2d(input_size=DEFAULT_INPUT_SIZE, input_tensor=None, output_mode=None, num_filters=None,
                     depth=6, dropout_rate=0.0, layer_order=['relu', 'bn', 'dropout', 'conv'],
                     use_squeeze_excitation=False, squeeze_excitation_ratio=8):
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
    pool = inputs
    for depth_cnt in range(depth):
        conv = Conv2D(nfeatures[depth_cnt], kernel_size=(3, 3), padding='same',
                      kernel_initializer=he_normal(seed=SEED),
                      name='enc_%d_input' % (depth_cnt + 1))(pool)

        conv = res_block(conv, base_name='enc_%d' % (depth_cnt + 1), nfeatures=nfeatures[depth_cnt],
                         layer_order=layer_order,
                         dropout_rate=dropout_rate,
                         seed=SEED,
                         use_squeeze_excitation=use_squeeze_excitation,
                         squeeze_excitation_ratio=squeeze_excitation_ratio)

        conv = Activation('relu', name='enc_%d_ouput_relu' % (depth_cnt + 1))(conv)
        conv = BN(axis=-1, momentum=0.95, epsilon=0.001, name='enc_%d_ouput_bn' % (depth_cnt + 1))(conv)

        conv_ptr.append(conv)

        # Only maxpool till penultimate depth
        if depth_cnt < depth - 1:

            # If size of input is odd, only do a 3x3 max pool
            xres = conv.shape.as_list()[1]
            if (xres % 2 == 0):
                pooling_size = (2, 2)
            elif (xres % 2 == 1):
                pooling_size = (3, 3)

            pool = MaxPooling2D(pool_size=pooling_size, name='enc_maxpool_%d' % (depth_cnt + 1))(conv)

    # step up convolutional layers
    for depth_cnt in range(depth - 2, -1, -1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # If size of input is odd, then do a 3x3 deconv
        if (deconv_shape[1] % 2 == 0):
            unpooling_size = (2, 2)
        elif (deconv_shape[1] % 2 == 1):
            unpooling_size = (3, 3)

        up = Concatenate(axis=3)([Conv2DTranspose(nfeatures[depth_cnt], (3, 3),
                                                  padding='same',
                                                  strides=unpooling_size,
                                                  kernel_initializer=he_normal(seed=SEED),
                                                  name='dec_deconv_%d' % (depth_cnt + 1))(conv),
                                  conv_ptr[depth_cnt]])

        up = Conv2D(nfeatures[depth_cnt], kernel_size=(3, 3), padding='same',
                    kernel_initializer=he_normal(seed=SEED),
                    name='dec_%d_input' % (depth_cnt + 1))(up)

        conv = res_block(up, base_name='dec_%d' % (depth_cnt + 1), nfeatures=nfeatures[depth_cnt], seed=SEED,
                         layer_order=layer_order,
                         use_squeeze_excitation=use_squeeze_excitation,
                         squeeze_excitation_ratio=squeeze_excitation_ratio)

        conv = Activation('relu', name='dec_%d_ouput_relu' % (depth_cnt + 1))(conv)
        conv = BN(axis=-1, momentum=0.95, epsilon=0.001, name='dec_%d_ouput_bn' % (depth_cnt + 1))(conv)

    if output_mode is not None:
        recon = Conv2D(1, (1, 1), padding='same',
                       kernel_initializer=he_normal(seed=SEED),
                       name='1d_class_conv')(conv)
        recon = Activation(output_mode, name='class_%s' % output_mode)(recon)
    else:
        recon = conv

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs=inputs, outputs=[recon])

    return model


if __name__ == '__main__':
    save_path = '../imgs/res_unet2d_se.png'
    m = residual_unet_2d(use_squeeze_excitation=True)
    plot_model(m, to_file=save_path)
