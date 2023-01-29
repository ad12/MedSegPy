"""Convolutional Layers

The following layers are possible replacements for common convolutional
layers.
"""
import tensorflow as tf
import keras.backend as K

from keras.layers.convolutional import Conv2D


class ConvStandardized2D(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(ConvStandardized2D, self).__init__(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def call(self, inputs):
        """
        This call function is essentially a copy of the call function
        for the private class _Conv, taking only the parts relevant
        for 2D convolution, and adding weight standardization at the
        very beginning of the function.
        """
        # Weight standardization
        # -- copied from "https://github.com/joe-siyuan-qiao/WeightStandardization"
        kernel = self.kernel
        kernel_mean = tf.math.reduce_mean(kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
        kernel = kernel - kernel_mean
        kernel_std = K.std(kernel, axis=[0, 1, 2], keepdims=True)
        kernel = kernel / (kernel_std + 1e-5)

        # Copied from call() in _Conv class
        outputs = K.conv2d(
            inputs,
            kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
