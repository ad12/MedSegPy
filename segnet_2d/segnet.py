# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.utils import plot_model
from keras.initializers import glorot_uniform, he_normal
from segnet_2d.Mylayers import MaxPoolingWithArgmax2D, MaxUnpooling2D

import glob_constants as glc

def _encoder_block(x, level, num_conv_layers=2, num_filters=64, kernel=3, pool_size=(2,2), single_bn=False):
    if (num_conv_layers <= 0):
        raise ValueError('Must have at least 1 conv layer')

    curr_layer = x

    for i in range(num_conv_layers):
        conv = Convolution2D(num_filters,
                             (kernel, kernel),
                             padding="same",
                             kernel_initializer=he_normal(seed=glc.SEED),
                             name='enc_%d_conv_%d' % (level, i+1))(curr_layer)
        if not single_bn:
            conv = BatchNormalization(name='enc_%d_bn_%d' % (level, i+1))(conv)
        conv = Activation("relu", name='enc_%d_relu_%d' % (level, i+1))(conv)
        curr_layer = conv

    if single_bn:
        curr_layer = BatchNormalization(name='enc_%d_bn' % level)(curr_layer)

    l_pool, l_mask = MaxPoolingWithArgmax2D(pool_size=pool_size, strides=pool_size, name='enc_%d_pool' % level )(curr_layer)

    return (l_pool, l_mask)


def _decoder_block(x_pool, x_mask, level, num_conv_layers=2, num_filters=64, num_filters_next=32, kernel=3, pool_size=(2,2), single_bn=False):

    unpool_1 = MaxUnpooling2D(pool_size)([x_pool, x_mask])

    curr_layer = unpool_1

    for i in range(num_conv_layers):
        used_num_filters = num_filters_next if i==num_conv_layers-1 else num_filters
        conv = Convolution2D(used_num_filters,
                             (kernel, kernel),
                             kernel_initializer=he_normal(seed=glc.SEED),
                             padding="same",
                             name='dec_%d_conv_%d' % (level, i+1))(curr_layer)

        if not single_bn:
            conv = BatchNormalization(name='dec_%d_bn_%d' % (level, i+1))(conv)

        conv = Activation("relu", name='dec_%d_relu_%d' % (level, i+1))(conv)
        curr_layer = conv

    if single_bn:
        curr_layer = BatchNormalization(name='dec_%d_bn' % level)(curr_layer)

    return curr_layer


def _encoder_block_conv_act_bn(x, level, num_conv_layers=2, num_filters=64, kernel=3, pool_size=(2,2)):
    if (num_conv_layers <= 0):
        raise ValueError('Must have at least 1 conv layer')

    curr_layer = x

    for i in range(num_conv_layers):
        conv = Convolution2D(num_filters,
                             (kernel, kernel),
                             padding="same",
                             kernel_initializer=he_normal(seed=glc.SEED),
                             name='enc_%d_conv_%d' % (level, i+1))(curr_layer)
        conv = Activation("relu",
                          name='enc_%d_relu_%d' % (level, i+1))(conv)
        conv = BatchNormalization(name='enc_%d_bn_%d' % (level, i+1))(conv)
        curr_layer = conv

    l_pool, l_mask = MaxPoolingWithArgmax2D(pool_size=pool_size, strides=pool_size, name='enc_%d_pool' % level )(curr_layer)

    return (l_pool, l_mask)


def _decoder_block_conv_act_bn(x_pool, x_mask, level, num_conv_layers=2, num_filters=64, num_filters_next=32, kernel=3, pool_size=(2,2)):

    unpool_1 = MaxUnpooling2D(pool_size)([x_pool, x_mask])

    curr_layer = unpool_1

    for i in range(num_conv_layers):
        used_num_filters = num_filters_next if i==num_conv_layers-1 else num_filters
        conv = Convolution2D(used_num_filters,
                             (kernel, kernel),
                             padding="same",
                             kernel_initializer=he_normal(seed=glc.SEED),
                             name='dec_%d_conv_%d' % (level, i+1))(curr_layer)
        conv = Activation("relu", name='dec_%d_relu_%d' % (level, i+1))(conv)
        conv = BatchNormalization(name='dec_%d_bn_%d' % (level, i + 1))(conv)
        curr_layer = conv

    return curr_layer

def Segnet_v2(input_shape=(288,288,1), input_tensor=None, n_labels=1, depth=5, num_conv_layers=[2, 2, 3, 3, 3], num_filters=[64, 128, 256, 512, 512], kernel=3, pool_size=(2, 2), output_mode="sigmoid", single_bn=False, conv_act_bn=False):

    print('Initializing segnet with seed: %s' % str(glc.SEED))

    if input_tensor is not None:
        inputs = input_tensor
    else:
        inputs = Input(shape=input_shape)

    mask_layers = []

    curr_layer = inputs
    eff_pool_sizes = []
    
    # Determine pool sizes
    for i in range(depth):
        level = i+1
        eff_pool_size = pool_size
        divisor = pool_size[0] ** level
        if (input_shape[0] % divisor != 0):
            eff_pool_size = (3,3)
        eff_pool_sizes.append(eff_pool_size)

    # encoder
    print('Building Encoder...')
    for i in range(depth):
        eff_pool_size = eff_pool_sizes[i]
        if (conv_act_bn):
            curr_layer, l_mask = _encoder_block_conv_act_bn(curr_layer,
                                                            level=i + 1,
                                                            num_conv_layers=num_conv_layers[i],
                                                            num_filters=num_filters[i],
                                                            kernel=kernel,
                                                            pool_size=eff_pool_size)
        else:
            curr_layer, l_mask = _encoder_block(curr_layer,
                                                level=i+1,
                                                num_conv_layers=num_conv_layers[i],
                                                num_filters=num_filters[i],
                                                kernel=kernel,
                                                pool_size=eff_pool_size,
                                                single_bn=single_bn)

        mask_layers.append(l_mask)

    print('Building decoder...')
    # decoder
    for i in reversed(range(depth)):
        l_mask = mask_layers[i]
        eff_pool_size = eff_pool_sizes[i]
        if (conv_act_bn):
            curr_layer = _decoder_block_conv_act_bn(curr_layer,
                                                    l_mask,
                                                    level=i+1,
                                                    num_conv_layers=num_conv_layers[i],
                                                    num_filters=num_filters[i],
                                                    num_filters_next = 1 if i==0 else num_filters[i-1],
                                                    kernel=kernel,
                                                    pool_size=eff_pool_size)
        else:
            curr_layer = _decoder_block(curr_layer,
                                        l_mask,
                                        level=i+1,
                                        num_conv_layers=num_conv_layers[i],
                                        num_filters=num_filters[i],
                                        num_filters_next = 1 if i==0 else num_filters[i-1],
                                        kernel=kernel,
                                        pool_size=eff_pool_size,
                                        single_bn=single_bn)

    outputs = Convolution2D(n_labels,
                            (1, 1),
                            kernel_initializer=glorot_uniform(seed=glc.SEED),
                            activation=output_mode)(curr_layer)

    segnet = Model(inputs=inputs, outputs=outputs, name="segnet")

    return segnet


def Segnet(input_shape=(288,288,1), input_tensor=None, n_labels=1, kernel=3, pool_size=(2, 2), output_mode="sigmoid"):
    # encoder
    if (input_tensor is not None):
        inputs = input_tensor
    else:
        inputs = Input(shape=input_shape)
        print(input_shape)

    conv_1 = Convolution2D(64, (kernel, kernel), padding="same")(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation("relu")(conv_1)
    conv_2 = Convolution2D(64, (kernel, kernel), padding="same")(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation("relu")(conv_2)

    pool_1, mask_1 = MaxPoolingWithArgmax2D(pool_size)(conv_2)

    conv_3 = Convolution2D(128, (kernel, kernel), padding="same")(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation("relu")(conv_3)
    conv_4 = Convolution2D(128, (kernel, kernel), padding="same")(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation("relu")(conv_4)

    pool_2, mask_2 = MaxPoolingWithArgmax2D(pool_size)(conv_4)

    conv_5 = Convolution2D(256, (kernel, kernel), padding="same")(pool_2)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation("relu")(conv_5)
    conv_6 = Convolution2D(256, (kernel, kernel), padding="same")(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation("relu")(conv_6)
    conv_7 = Convolution2D(256, (kernel, kernel), padding="same")(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation("relu")(conv_7)

    pool_3, mask_3 = MaxPoolingWithArgmax2D(pool_size)(conv_7)

    conv_8 = Convolution2D(512, (kernel, kernel), padding="same")(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation("relu")(conv_8)
    conv_9 = Convolution2D(512, (kernel, kernel), padding="same")(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation("relu")(conv_9)
    conv_10 = Convolution2D(512, (kernel, kernel), padding="same")(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation("relu")(conv_10)

    pool_4, mask_4 = MaxPoolingWithArgmax2D(pool_size)(conv_10)

    conv_11 = Convolution2D(512, (kernel, kernel), padding="same")(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation("relu")(conv_11)
    conv_12 = Convolution2D(512, (kernel, kernel), padding="same")(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation("relu")(conv_12)
    conv_13 = Convolution2D(512, (kernel, kernel), padding="same")(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation("relu")(conv_13)

    pool_5, mask_5 = MaxPoolingWithArgmax2D(pool_size)(conv_13)
    print("Build encoder done..")

    # decoder

    unpool_1 = MaxUnpooling2D(pool_size)([pool_5, mask_5])

    conv_14 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation("relu")(conv_14)
    conv_15 = Convolution2D(512, (kernel, kernel), padding="same")(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation("relu")(conv_15)
    conv_16 = Convolution2D(512, (kernel, kernel), padding="same")(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation("relu")(conv_16)

    unpool_2 = MaxUnpooling2D(pool_size)([conv_16, mask_4])

    conv_17 = Convolution2D(512, (kernel, kernel), padding="same")(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation("relu")(conv_17)
    conv_18 = Convolution2D(512, (kernel, kernel), padding="same")(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation("relu")(conv_18)
    conv_19 = Convolution2D(256, (kernel, kernel), padding="same")(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation("relu")(conv_19)

    unpool_3 = MaxUnpooling2D(pool_size)([conv_19, mask_3])

    conv_20 = Convolution2D(256, (kernel, kernel), padding="same")(unpool_3)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation("relu")(conv_20)
    conv_21 = Convolution2D(256, (kernel, kernel), padding="same")(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation("relu")(conv_21)
    conv_22 = Convolution2D(128, (kernel, kernel), padding="same")(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation("relu")(conv_22)

    unpool_4 = MaxUnpooling2D(pool_size)([conv_22, mask_2])

    conv_23 = Convolution2D(128, (kernel, kernel), padding="same")(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation("relu")(conv_23)
    conv_24 = Convolution2D(64, (kernel, kernel), padding="same")(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation("relu")(conv_24)

    unpool_5 = MaxUnpooling2D(pool_size)([conv_24, mask_1])

    conv_25 = Convolution2D(64, (kernel, kernel), padding="same")(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation("relu")(conv_25)


    conv_26 = Convolution2D(n_labels, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)
    conv_26 = Activation("relu")(conv_26)
    outputs = Convolution2D(n_labels, (1,1), activation=output_mode)(conv_26)
    print("Build decoder done..")

    segnet = Model(inputs=inputs, outputs=outputs, name="segnet")

    return segnet

if __name__ == '__main__':
    model = Segnet_v2(input_shape=(288, 288, 1), input_tensor=None, n_labels=1, depth=6, num_conv_layers=[2, 2, 3, 3, 3, 3],
              num_filters=[64, 128, 256, 512, 512, 512], kernel=3, pool_size=(2, 2), output_mode="sigmoid")

    plot_model(model, to_file='%s.png' % model.name, show_shapes=True)
