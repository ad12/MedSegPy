# -*- coding: utf-8 -*-

""" Deeplabv3+ model for Keras.
This model is based on TF repo:
https://github.com/tensorflow/models/tree/master/research/deeplab
On Pascal VOC, original model gets to 84.56% mIOU

Now this model is only available for the TensorFlow backend,
due to its reliance on `SeparableConvolution` layers, but Theano will add
this layer soon.

MobileNetv2 backbone is based on this repo:
https://github.com/JonathanCMitchell/mobilenet_v2_keras

# Reference
- [Encoder-Decoder with Atrous Separable Convolution
    for Semantic Image Segmentation](https://arxiv.org/pdf/1802.02611.pdf)
- [Xception: Deep Learning with Depthwise Separable Convolutions]
    (https://arxiv.org/abs/1610.02357)
- [Inverted Residuals and Linear Bottlenecks: Mobile Networks for
    Classification, Detection and Segmentation](https://arxiv.org/abs/1801.04381)
"""
import logging

from keras import backend as K
from keras.engine import InputSpec, Layer

from keras.utils import conv_utils


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(
        self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs
    ):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(
                output_size, 2, "output_size"
            )
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(
                upsampling, 2, "upsampling"
            )

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = (
                self.upsampling[0] * input_shape[1]
                if input_shape[1] is not None
                else None
            )
            width = (
                self.upsampling[1] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0], height, width, input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(
                inputs,
                (
                    inputs.shape[1] * self.upsampling[0],
                    inputs.shape[2] * self.upsampling[1],
                ),
                align_corners=True,
            )
        else:
            return K.tf.image.resize_bilinear(
                inputs,
                (self.output_size[0], self.output_size[1]),
                align_corners=True,
            )

    def get_config(self):
        config = {
            "upsampling": self.upsampling,
            "output_size": self.output_size,
            "data_format": self.data_format,
        }
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
