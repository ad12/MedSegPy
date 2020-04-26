"""Pooling layers.

MaxPoolingMask2D, MaxPoolingWithArgmax2D, and MaxUnpooling2D are based on are
based on code from the open-source repo below. We thank the authors for making
the code publicly available.

https://github.com/ykamikawa/tf-keras-SegNet/tree/648ee1aa6870e8280a5f24ee193caa585adde9cd.
"""

from keras import backend as K
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Layer
from keras.layers.pooling import MaxPooling2D

__all__ = ["MaxPoolingWithArgmax2D", "MaxUnpooling2D"]


class MaxPoolingWithArgmax2D(Layer):
    def __init__(
        self, pool_size=(2, 2), strides=(2, 2), padding="same", **kwargs
    ):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.padding = padding
        self.pool_size = pool_size
        self.strides = strides

    def call(self, inputs, **kwargs):
        padding = self.padding
        pool_size = self.pool_size
        strides = self.strides
        if K.backend() == "tensorflow":
            ksize = [1, pool_size[0], pool_size[1], 1]
            padding = padding.upper()
            strides = [1, strides[0], strides[1], 1]
            output, argmax = K.tf.nn.max_pool_with_argmax(
                inputs, ksize=ksize, strides=strides, padding=padding
            )
        else:
            errmsg = "{} backend is not supported for layer {}".format(
                K.backend(), type(self).__name__
            )
            raise NotImplementedError(errmsg)
        argmax = K.cast(argmax, K.floatx())
        return [output, argmax]

    def compute_output_shape(self, input_shape):
        ratio = (1, self.pool_size[0], self.pool_size[1], 1)
        output_shape = [
            dim // ratio[idx] if dim is not None else None
            for idx, dim in enumerate(input_shape)
        ]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return self.pool_size[0] * [None]

    def get_config(self):
        base_cfg = super().get_config()
        base_cfg.update(
            {
                "padding": self.padding,
                "pool_size": self.pool_size,
                "strides": self.strides,
            }
        )
        return base_cfg


class MaxUnpooling2D(Layer):
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = size

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        with K.tf.variable_scope(self.name):
            mask = K.cast(mask, "int32")
            input_shape = K.tf.shape(updates, out_type="int32")
            #  calculation new shape
            if output_shape is None:
                output_shape = (
                    input_shape[0],
                    input_shape[1] * self.pool_size[0],
                    input_shape[2] * self.pool_size[1],
                    input_shape[3],
                )
            self.output_shape1 = output_shape

            # calculation indices for batch, height, width and feature maps
            one_like_mask = K.ones_like(mask, dtype="int32")
            batch_shape = K.concatenate(
                [[input_shape[0]], [1], [1], [1]], axis=0
            )
            batch_range = K.reshape(
                K.tf.range(output_shape[0], dtype="int32"), shape=batch_shape
            )
            b = one_like_mask * batch_range
            y = mask // (output_shape[2] * output_shape[3])
            x = (mask // output_shape[3]) % output_shape[2]
            feature_range = K.tf.range(output_shape[3], dtype="int32")
            f = one_like_mask * feature_range

            # transpose indices & reshape update values to one dimension
            updates_size = K.tf.size(updates)
            indices = K.transpose(
                K.reshape(K.stack([b, y, x, f]), [4, updates_size])
            )
            values = K.reshape(updates, [updates_size])
            ret = K.tf.scatter_nd(indices, values, output_shape)
            return ret

    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        return (
            mask_shape[0],
            mask_shape[1] * self.pool_size[0],
            mask_shape[2] * self.pool_size[1],
            mask_shape[3],
        )

    def get_config(self):
        base_cfg = super().get_config()
        base_cfg.update({"pool_size": self.pool_size})
        return base_cfg