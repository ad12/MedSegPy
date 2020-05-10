from typing import Dict, Sequence, Union

from keras import backend as K
from keras.layers import (
    Concatenate,
    Conv2D,
    Conv3D,
    UpSampling2D,
    UpSampling3D,
    Multiply
)
from keras.layers import BatchNormalization as BN
from keras.layers import Layer
import numpy as np


class _CreateGatingSignalNDim(Layer):
    def __init__(self,
                 dimension: int,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]],
                 kernel_initializer: Union[str, Dict],
                 activation: str,
                 add_batchnorm: bool,
                 **kwargs
                 ):
        super(_CreateGatingSignalNDim, self).__init__(**kwargs)

        # Store parameters
        self.dimension = dimension
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.activation = activation
        self.add_batchnorm = add_batchnorm

        if self.dimension == 2:
            conv_type = Conv2D
        elif self.dimension == 3:
            conv_type = Conv3D
        else:
            raise ValueError("Only 2D and 3D are supported")

        if isinstance(self.kernel_size, tuple) or \
                isinstance(self.kernel_size, list):
            assert len(self.kernel_size) == self.dimension, \
                "If list/tuple, kernel_size must have length %d" % self.dimension

        self.conv = conv_type(
            out_channels,
            kernel_size,
            padding='same',
            activation=activation,
            kernel_initializer=kernel_initializer
        )
        if self.add_batchnorm:
            self.bn = BN(axis=-1, momentum=0.95, epsilon=0.001)

    def build(self, input_shape):
        self.conv.build(input_shape)
        self._trainable_weights = self.conv.trainable_weights
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.bn.build(conv_output_shape)
        self._trainable_weights += self.bn.trainable_weights
        super(_CreateGatingSignalNDim, self).build(input_shape)

    def call(self, inputs):
        outputs = self.conv(inputs)
        if self.add_batchnorm:
            outputs = self.bn(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        bn_output_shape = self.bn.compute_output_shape(conv_output_shape)
        return bn_output_shape

    def get_config(self):
        base_cfg = super().get_config()
        base_cfg.update(
            {
                'dimension': self.dimension,
                'out_channels': self.out_channels,
                'kernel_size': self.kernel_size,
                'kernel_initializer': self.kernel_initializer,
                'activation': self.activation,
                'add_batchnorm': self.add_batchnorm
            }
        )
        return base_cfg


class CreateGatingSignal2D(_CreateGatingSignalNDim):
    def __init__(self,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]] = 1,
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 activation: str = "relu",
                 add_batchnorm: bool = True,
                 **kwargs
                 ):
        super(CreateGatingSignal2D, self).__init__(
            dimension=2,
            out_channels=out_channels,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            activation=activation,
            add_batchnorm=add_batchnorm,
            **kwargs
        )


class CreateGatingSignal3D(_CreateGatingSignalNDim):
    def __init__(self,
                 out_channels: int,
                 kernel_size: Union[int, Sequence[int]] = 1,
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 activation: str = "relu",
                 add_batchnorm: bool = True,
                 **kwargs
                 ):
        super(CreateGatingSignal3D, self).__init__(
            dimension=3,
            out_channels=out_channels,
            kernel_size=kernel_size,
            kernel_initializer=kernel_initializer,
            activation=activation,
            add_batchnorm=add_batchnorm,
            **kwargs
        )


class _GridAttentionModuleND(Layer):
    def __init__(self,
                 dimension: int,
                 in_channels: int,
                 intermediate_channels: int,
                 sub_sample_factor: Union[int, Sequence[int]],
                 kernel_initializer: Union[str, Dict],
                 **kwargs
                 ):
        super(_GridAttentionModuleND, self).__init__(**kwargs)

        # Store parameters
        self.dimension = dimension
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.sub_sample_factor = sub_sample_factor
        self.kernel_initializer = kernel_initializer

        if self.dimension == 2:
            self.conv_type = Conv2D
            self.upsample_type = UpSampling2D
        elif self.dimension == 3:
            self.conv_type = Conv3D
            self.upsample_type = UpSampling3D
        else:
            raise ValueError("Only 2D and 3D are supported")

        if isinstance(self.sub_sample_factor, tuple) or \
                isinstance(self.sub_sample_factor, list):
            assert len(self.sub_sample_factor) == self.dimension, \
                "If list/tuple, sub_sample_factor must have length %d" % self.dimension

        self.theta_x = self.conv_type(
            self.intermediate_channels,
            kernel_size=self.sub_sample_factor,
            strides=self.sub_sample_factor,
            use_bias=False,
            kernel_initializer=self.kernel_initializer
        )

        self.theta_gating = self.conv_type(
            self.intermediate_channels,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer
        )

        self.phi = self.conv_type(
            1,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer
        )

        # Output Transform: Conv -> BN
        self.output_conv = self.conv_type(
            self.in_channels,
            kernel_size=1,
            kernel_initializer=self.kernel_initializer
        )

        self.output_bn = BN(axis=-1, momentum=0.95, epsilon=0.001)

    def build(self, input_shape):
        x_shape, gating_signal_shape = input_shape

        # Build theta_x
        self.theta_x.build(x_shape)
        self._trainable_weights = self.theta_x.trainable_weights
        theta_x_output_shape = self.theta_x.compute_output_shape(x_shape)

        # Build theta_gating
        self.theta_gating.build(gating_signal_shape)
        self._trainable_weights += self.theta_gating.trainable_weights
        theta_gating_output_shape = self.theta_gating.compute_output_shape(
            gating_signal_shape
        )

        # Build upsample_gating
        up_ratio_gating = np.divide(
            theta_x_output_shape[1:-1], theta_gating_output_shape[1:-1]
        )
        self.upsample_gating = self.upsample_type(
            size=up_ratio_gating
        )
        self.upsample_gating.build(theta_gating_output_shape)
        self._trainable_weights += self.upsample_gating.trainable_weights

        # Build phi
        self.phi.build(theta_x_output_shape)
        self._trainable_weights += self.phi.trainable_weights
        phi_output_shape = self.phi.compute_output_shape(
            theta_x_output_shape
        )

        # Build upsample_attn_coeff
        up_ratio_attn_coeff = np.divide(
            x_shape[1:-1], phi_output_shape[1:-1]
        )
        self.upsample_attn_coeff = self.upsample_type(
            size=up_ratio_attn_coeff
        )
        self.upsample_attn_coeff.build(phi_output_shape)
        self._trainable_weights += self.upsample_attn_coeff.trainable_weights

        # Build output_conv
        self.output_conv.build(x_shape)
        self._trainable_weights += self.output_conv.trainable_weights
        output_conv_output_shape = self.output_conv.compute_output_shape(
            x_shape
        )

        # Build output_bn
        self.output_bn.build(output_conv_output_shape)
        self._trainable_weights += self.output_bn.trainable_weights

        super(_GridAttentionModuleND, self).build(input_shape)

    def call(self, inputs):
        x, gating_signal = inputs
        theta_x_out = self.theta_x(x)
        theta_gating_out = self.theta_gating(gating_signal)

        # If theta_gating_out is smaller than theta_x_out,
        # then upsample using Upsample2D or Upsample3D.
        # NOTE: There does not exist a trilinear interpolation
        # mode for UpSample3D. We may need to make one
        # ourselves or use transposed convolution and learn the
        # upsampling operation.
        up_sampled_gating = self.upsample_gating(theta_gating_out)
        phi_out = self.phi(
            K.relu(theta_x_out + up_sampled_gating)
        )
        sigmoid_phi = K.sigmoid(phi_out)
        x_size = K.int_shape(x)

        # Need to upsample to size of inputs, such that
        # attention coefficients can be multiplied with inputs
        up_sampled_attn_coeff = self.upsample_attn_coeff(sigmoid_phi)
        attn_weighted_output = Multiply()([
            K.repeat_elements(up_sampled_attn_coeff,
                              rep=x_size[-1],
                              axis=-1),
            x]
        )
        output = self.output_conv(attn_weighted_output)
        output = self.output_bn(output)

        return output

    def compute_output_shape(self, input_shape):
        x_shape, gating_signal_shape = input_shape

        # Output shape for output_conv
        output_conv_output_shape = self.output_conv.compute_output_shape(
            x_shape
        )

        # Output shape for output_bn
        output_bn_output_shape = self.output_bn.compute_output_shape(
            output_conv_output_shape
        )
        return output_bn_output_shape

    def get_config(self):
        base_cfg = super().get_config()
        base_cfg.update(
            {
                "dimension": self.dimension,
                "in_channels": self.in_channels,
                "intermediate_channels": self.intermediate_channels,
                "sub_sample_factor": self.sub_sample_factor,
                "kernel_initializer": self.kernel_initializer
            }
        )
        return base_cfg


class GridAttentionModule2D(_GridAttentionModuleND):
    def __init__(self,
                 in_channels: int,
                 intermediate_channels: int,
                 sub_sample_factor: Union[int, Sequence[int]] = 2,
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 **kwargs
                 ):
        super(GridAttentionModule2D, self).__init__(
            dimension=2,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            sub_sample_factor=sub_sample_factor,
            kernel_initializer=kernel_initializer,
            **kwargs
        )


class GridAttentionModule3D(_GridAttentionModuleND):
    def __init__(self,
                 in_channels: int,
                 intermediate_channels: int,
                 sub_sample_factor: Union[int, Sequence[int]] = 2,
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 **kwargs
                 ):
        super(GridAttentionModule3D, self).__init__(
            dimension=3,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            sub_sample_factor=sub_sample_factor,
            kernel_initializer=kernel_initializer,
            **kwargs
        )


class _MultiAttentionModuleND(Layer):
    def __init__(self,
                 dimension: int,
                 in_channels: int,
                 intermediate_channels: int,
                 sub_sample_factor: Union[int, Sequence[int]],
                 kernel_initializer: Union[str, Dict],
                 activation: str,
                 **kwargs
                 ):
        super(_MultiAttentionModuleND, self).__init__(**kwargs)

        # Store parameters
        self.dimension = dimension
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.sub_sample_factor = sub_sample_factor
        self.kernel_initializer = kernel_initializer

        if self.dimension == 2:
            self.conv_type = Conv2D
            self.attn_module_type = GridAttentionModule2D
        elif self.dimension == 3:
            self.conv_type = Conv3D
            self.attn_module_type = GridAttentionModule3D
        else:
            raise ValueError("Only 2D and 3D are supported")

        self.attn_gate_1 = self.attn_module_type(
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            sub_sample_factor=sub_sample_factor,
            kernel_initializer=kernel_initializer
        )

        self.attn_gate_2 = self.attn_module_type(
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            sub_sample_factor=sub_sample_factor,
            kernel_initializer=kernel_initializer
        )

        self.combine_gates_conv = self.conv_type(
            in_channels,
            kernel_size=1,
            activation=activation,
            kernel_initializer=kernel_initializer
        )
        self.combine_gates_bn = BN(axis=-1, momentum=0.95, epsilon=0.001)

    def build(self, input_shape):
        # Build attention gates
        self.attn_gate_1.build(input_shape)
        self._trainable_weights = self.attn_gate_1.trainable_weights
        self.attn_gate_2.build(input_shape)
        self._trainable_weights += self.attn_gate_2.trainable_weights
        output_attn_shape = self.attn_gate_2.compute_output_shape(input_shape)

        concatenate_gate_shape = list(output_attn_shape)
        concatenate_gate_shape[-1] *= 2
        self.combine_gates_conv.build(concatenate_gate_shape)
        self._trainable_weights += self.combine_gates_conv.trainable_weights
        conv_output_shape = self.combine_gates_conv.compute_output_shape(
            concatenate_gate_shape
        )
        self.combine_gates_bn.build(conv_output_shape)
        self._trainable_weights += self.combine_gates_bn.trainable_weights
        super(_MultiAttentionModuleND, self).build(input_shape)

    def call(self, inputs):
        x, gating_signal = inputs
        gate_1_out = self.attn_gate_1([x, gating_signal])
        gate_2_out = self.attn_gate_2([x, gating_signal])

        total_gate_outputs = Concatenate(axis=-1)([gate_1_out, gate_2_out])
        output = self.combine_gates_conv(total_gate_outputs)
        output = self.combine_gates_bn(output)
        return output

    def compute_output_shape(self, input_shape):
        output_attn_shape = self.attn_gate_2.compute_output_shape(
            input_shape
        )
        concatenate_gate_shape = list(output_attn_shape)
        concatenate_gate_shape[-1] *= 2
        conv_output_shape = self.combine_gates_conv.compute_output_shape(
            concatenate_gate_shape
        )
        bn_output_shape = self.combine_gates_bn.compute_output_shape(
            conv_output_shape
        )
        return bn_output_shape

    def get_config(self):
        base_cfg = super().get_config()
        base_cfg.update(
            {
                'dimension': self.dimension,
                'in_channels': self.in_channels,
                'intermediate_channels': self.intermediate_channels,
                'sub_sample_factor': self.sub_sample_factor,
                'kernel_initializer': self.kernel_initializer
            }
        )
        return base_cfg


class MultiAttentionModule2D(_MultiAttentionModuleND):
    def __init__(self,
                 in_channels: int,
                 intermediate_channels: int,
                 sub_sample_factor: Union[int, Sequence[int]] = 2,
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 activation: str = "relu",
                 **kwargs
                 ):
        super(MultiAttentionModule2D, self).__init__(
            dimension=2,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            sub_sample_factor=sub_sample_factor,
            kernel_initializer=kernel_initializer,
            activation=activation,
            **kwargs
        )


class MultiAttentionModule3D(_MultiAttentionModuleND):
    def __init__(self,
                 in_channels: int,
                 intermediate_channels: int,
                 sub_sample_factor: Union[int, Sequence[int]] = 2,
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 activation: str = "relu",
                 **kwargs
                 ):
        super(MultiAttentionModule3D, self).__init__(
            dimension=3,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            sub_sample_factor=sub_sample_factor,
            kernel_initializer=kernel_initializer,
            activation=activation,
            **kwargs
        )


class _DeepSupervisionND(Layer):
    def __init_(self,
                dimension: int,
                out_channels: int,
                scale_factor: Union[int, Sequence[int]],
                kernel_initializer: Union[str, Dict],
                **kwargs
                ):
        super(_DeepSupervisionND, self).__init__(**kwargs)

        # Store parameters
        self.dimension = dimension
        self.out_channels = out_channels
        if isinstance(scale_factor, list):
            self.scale_factor = tuple(scale_factor)

        if self.dimension == 2:
            self.conv_type = Conv2D
            self.upsample = UpSampling2D(
                size=scale_factor
            )
        elif self.dimension == 3:
            self.conv_type = Conv3D
            self.upsample = UpSampling3D(
                size=scale_factor
            )
        else:
            raise ValueError("Only 2D and 3D are supported")

        self.conv = self.conv_type(
            out_channels,
            kernel_size=1,
            kernel_initializer=kernel_initializer
        )

    def build(self, input_shape):
        self.conv.build(input_shape)
        self._trainable_weights = self.conv.trainable_weights
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        self.upsample.build(conv_output_shape)
        self._trainable_weights += self.upsample.trainable_weights
        super(_DeepSupervisionND, self).build(input_shape)

    def call(self, inputs):
        outputs = self.conv(inputs)
        outputs = self.upsample(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        conv_output_shape = self.conv.compute_output_shape(input_shape)
        upsample_output_shape = self.upsample.compute_output_shape(
            conv_output_shape
        )
        return upsample_output_shape

    def get_config(self):
        base_cfg = super().get_config()
        base_cfg.update(
            {
                'dimension': self.dimension,
                'out_channels': self.out_channels,
                'scale_factor': self.scale_factor
            }
        )
        return base_cfg


class DeepSupervision2D(_DeepSupervisionND):
    def __init__(self,
                 out_channels: int,
                 scale_factor: Union[int, Sequence[int]],
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 **kwargs
                 ):
        super(DeepSupervision2D, self).__init__(
            dimension=2,
            out_channels=out_channels,
            scale_factor=scale_factor,
            kernel_initializer=kernel_initializer,
            **kwargs
        )


class DeepSupervision3D(_DeepSupervisionND):
    def __init__(self,
                 out_channels: int,
                 scale_factor: Union[int, Sequence[int]],
                 kernel_initializer: Union[str, Dict] = "he_normal",
                 **kwargs
                 ):
        super(DeepSupervision3D, self).__init__(
            dimension=3,
            out_channels=out_channels,
            scale_factor=scale_factor,
            kernel_initializer=kernel_initializer,
            **kwargs
        )
