# Implementation of refine module used by RefineNet
# (http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf)
import sys
from typing import Iterable, List
from keras.layers import Layer, Activation, Conv2D, Add, MaxPooling2D
from keras.layers import BatchNormalization as BN


sys.path.append('../../')
from models.deeplab_2d.deeplab_model import BilinearUpsampling


def refine_module(xs_in: List[Layer], num_filters_in: int, num_filters_out: int, name_prefix: str):
    # 1. adaptive convolutions
    adaptive_convs = []

    for i in range(len(xs_in)):
        x = xs_in[i]
        adaptive_convs.append(__residual_conv_unit__(x, num_filters_in, '%s_adp-conv%d' % (name_prefix, i)))

    # 2. multi-resolution fusion
    x = __multi_resolution_fusion__(adaptive_convs, name_prefix)

    # 3. chained residual pooling
    x = __chained_residual_pooling__(x, num_filters_in, name_prefix)
    x = __residual_conv_unit__(x, num_filters_in, name_prefix)

    x = Conv2D(filters=num_filters_out, kernel_size=(3,3),
               padding='same',
               name='%s_conv' % name_prefix)(x)
    x = BN(axis=-1, momentum=0.95, epsilon=0.001,
           name='%s_bn' % name_prefix)(x)
    x = Activation('relu', name='%s_relu' % name_prefix)(x)

    return x


def __residual_conv_unit__(x_in: Layer, num_filters: int, name_prefix: str):
    name_prefix = '%s_rcu' % name_prefix
    x = x_in
    for i in range(2):
        x = BN(axis=-1, momentum=0.95, epsilon=0.001, name='%s_bn_%d' % (name_prefix, i))(x)
        x = Activation('relu', name='%s_relu_%d' % (name_prefix, i))(x)
        x = Conv2D(filters=num_filters, kernel_size=(3,3),
                   padding='same',
                   name='%s_conv_%d' % (name_prefix, i))(x)

    x = Add(name='%s_add' % name_prefix)([x_in, x])
    return x


def __multi_resolution_fusion__(xs_in: List[Layer], name_prefix: str):
    name_prefix = '%s_mrf' % name_prefix
    if len(xs_in) == 1:
        return xs_in[0]

    input_shapes = []
    num_channels = []
    for x in xs_in:
        input_shapes.append(x.shape.as_list()[1:-1])
        num_channels.append(x.shape.as_list()[-1])

    # upsample all inputs to the largest input size
    # assume largest input size is larger in all dimensions
    max_input_shape = max(input_shapes, key=lambda x: x[0])
    largest_input_ind = input_shapes.index(max_input_shape)

    # smallest number of filters
    num_filters = min(num_channels)

    xs = []
    for i in range(len(xs_in)):
        x = xs_in[i]
        x = Conv2D(filters=num_filters, kernel_size=(3,3),
                   padding='same',
                   name='%s_conv_%d' % (name_prefix, i))(x)
        x = BN(axis=-1, momentum=0.95, epsilon=0.001,
               name='%s_bn_%d' % (name_prefix, i))(x)

        if i != largest_input_ind:
            x = BilinearUpsampling(output_size=max_input_shape,
                                   name='%s_blu_%d' % (name_prefix, i))(x)
        xs.append(x)

    x = Add(name='%s_mrf_add' % name_prefix)(xs)
    return x


def __chained_residual_pooling__(x_in: Layer, num_filters: int, name_prefix: str):
    name_prefix = '%s_crp' % name_prefix
    x = BN(axis=-1, momentum=0.95, epsilon=0.001, name='%s_bn' % name_prefix)(x_in)
    x = Activation('relu', name='%s_relu' % name_prefix)(x)
    residuals = [x]
    for i in range(3):
        x = MaxPooling2D(pool_size=(3,3), strides=1, padding='same', name='%s_pool_%d' % (name_prefix, i))(x)
        x = Conv2D(filters=num_filters, kernel_size=(3,3),
                   padding='same',
                   name='%s_conv_%d' % (name_prefix, i))(x)
        residuals.append(x)

    x = Add(name='%s_add' % name_prefix)(residuals)

    return x


