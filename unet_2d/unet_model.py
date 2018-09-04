from __future__ import print_function, division

import numpy as np
import os
import warnings
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, Concatenate
from keras.layers import BatchNormalization as BN
import keras.backend as K
from keras.engine.topology import get_source_inputs

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


def unet_2d_model(input_size=DEFAULT_INPUT_SIZE, input_tensor=None, output_mode="sigmoid"):
    """Generate Unet 2D model compatible with Keras 2

    :param input_size: tuple of input size - format: (height, width, 1)

    :rtype: Keras model

    :raise ValueError if input_size is not tuple or dimensions of input_size do not match (height, width, 1)
    """

    if input_tensor is None and (type(input_size) is not tuple or len(input_size) != 3):
        raise ValueError('input_size must be a tuple of size (height, width, 1)')

    nfeatures = [2 ** feat * 32 for feat in np.arange(6)]
    depth = len(nfeatures)

    conv_ptr = []

    # input layer
    inputs = input_tensor if input_tensor is not None else Input(input_size)

    # step down convolutional layers
    pool = inputs
    for depth_cnt in range(depth):

        conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                      padding='same',
                      activation='relu',
                      kernel_initializer='he_normal')(pool)
        conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                      padding='same',
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.0)(conv)

        conv_ptr.append(conv)

        # Only maxpool till penultimate depth
        if depth_cnt < depth - 1:

            # If size of input is odd, only do a 3x3 max pool
            xres = conv.shape.as_list()[1]
            if (xres % 2 == 0):
                pooling_size = (2, 2)
            elif (xres % 2 == 1):
                pooling_size = (3, 3)

            pool = MaxPooling2D(pool_size=pooling_size)(conv)

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
                                                  strides=unpooling_size)(conv),
                                 conv_ptr[depth_cnt]])

        conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                      padding='same',
                      activation='relu',
                      kernel_initializer='he_normal')(up)
        conv = Conv2D(nfeatures[depth_cnt], (3, 3),
                      padding='same',
                      activation='relu',
                      kernel_initializer='he_normal')(conv)

        conv = BN(axis=-1, momentum=0.95, epsilon=0.001)(conv)
        conv = Dropout(rate=0.00)(conv)

    # combine features
    recon = Conv2D(1, (1, 1), padding='same', activation=output_mode)(conv)

    if (input_tensor is not None):
        inputs = get_source_inputs(input_tensor)

    model = Model(inputs=inputs, outputs=[recon])

    return model
