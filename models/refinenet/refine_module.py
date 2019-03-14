# Implementation of refine module used by RefineNet
# (http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_RefineNet_Multi-Path_Refinement_CVPR_2017_paper.pdf)
import sys
from typing import Iterable, List
from keras.layers import Layer, Activation, Conv2D, Add, MaxPooling2D
from keras.layers import BatchNormalization as BN


sys.path.append('../../')
from models.deeplab_2d.deeplab_model import BilinearUpsampling


def refine_module(xs_in: List[Layer], num_filters_in: int, num_filters_out: int = None):
    if not num_filters_out:
        num_filters_out = num_filters_in

    # 1. adaptive convolutions
    adaptive_convs = []
    for x in xs_in:
        adaptive_convs.append(__residual_conv_unit__(x, num_filters_in))

    # 2. multi-resolution fusion
    x = __multi_resolution_fusion__(adaptive_convs)

    # 3. chained residual pooling
    x = __chained_residual_pooling__(x, num_filters_in)
    x = __residual_conv_unit__(x, num_filters_in)

    x = Conv2D(filters=num_filters_out, kernel_size=(3,3), padding='same')(x)
    x = BN(axis=-1, momentum=0.95, epsilon=0.001)(x)
    x = Activation('relu')(x)

    return x


def __residual_conv_unit__(x_in: Layer, num_filters: int):
    x = x_in
    for i in range(2):
        x = BN(axis=-1, momentum=0.95, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=num_filters, kernel_size=(3,3), padding='same')(x)

    x = Add()([x_in, x])
    return x


def __multi_resolution_fusion__(xs_in: List[Layer]):
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
        x = Conv2D(filters=num_filters, kernel_size=(3,3), padding='same')(x)
        x = BN(axis=-1, momentum=0.95, epsilon=0.001)(x)
        if i != largest_input_ind:
            x = BilinearUpsampling(output_size=max_input_shape)(x)
        xs.append(x)

    x = Add()(xs)
    return x


def __chained_residual_pooling__(x_in: Layer, num_filters: int):
    x = BN(axis=-1, momentum=0.95, epsilon=0.001)(x_in)
    x = Activation('relu')(x)
    residuals = [x]
    for i in range(3):
        x = MaxPooling2D(pool_size=(3,3), strides=1, padding='same')(x)
        x = Conv2D(filters=num_filters, kernel_size=(3,3),
                   padding='same')(x)
        residuals.append(x)

    x = Add()(residuals)

    return x


