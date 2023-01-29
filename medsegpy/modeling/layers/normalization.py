"""Normalization Layers

The following layers are useful for normalizing the inputs to
layers.

1) Group Normalization:
    - Copied from keras_contrib/layers/normalization/groupnormalization.py
    - Replaced the call to KC.moments with a call to the actual tensorflow function
    - Modified 'get_config()' to match structure of 'get_config()' in ./attention.py
"""
import tensorflow as tf

from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
from keras import backend as K


class GroupNormalization(Layer):
    """Group normalization layer.
    Group Normalization divides the channels into groups and computes
    within each group
    the mean and variance for normalization.
    Group Normalization's computation is independent
     of batch sizes, and its accuracy is stable in a wide range of batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical to
    Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        mean_var_shape = (self.groups,)

        self.moving_mean = self.add_weight(
            shape=mean_var_shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)

        self.moving_variance = self.add_weight(
            shape=mean_var_shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)

        self.built = True

    def call(self, inputs, training=None, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        def normalize_inference(inputs):
            moving_broadcast_shape = [1] * len(input_shape)
            moving_broadcast_shape.insert(1, self.groups)

            # Broadcast moving mean and moving variance
            broadcast_moving_mean = K.reshape(self.moving_mean,
                                              moving_broadcast_shape)
            broadcast_moving_variance = K.reshape(self.moving_variance,
                                                  moving_broadcast_shape)
            # Perform group normalization
            inputs = (inputs - broadcast_moving_mean) / (
                K.sqrt(broadcast_moving_variance + self.epsilon))

            # prepare broadcast shape
            inputs = K.reshape(inputs, group_shape)
            outputs = inputs

            # In this case we must explicitly broadcast all parameters.
            if self.scale:
                broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                outputs = outputs * broadcast_gamma

            if self.center:
                broadcast_beta = K.reshape(self.beta, broadcast_shape)
                outputs = outputs + broadcast_beta

            # finally we reshape the output back to the input shape
            outputs = K.reshape(outputs, tensor_input_shape)

            return outputs

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference(inputs)

        group_reduction_axes = list(range(len(group_axes)))
        mean, variance = tf.nn.moments(inputs,
                                       axes=group_reduction_axes[2:],
                                       keep_dims=True)
        normed_inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        batch_mean = K.mean(mean, axis=0, keepdims=True)
        batch_variance = K.mean(variance, axis=0, keepdims=True)

        moving_shape = K.shape(self.moving_mean)
        batch_mean = K.reshape(batch_mean, moving_shape)
        batch_variance = K.reshape(batch_variance, moving_shape)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(normed_inputs)[axis]
                                  for axis in group_reduction_axes[2:]])
            sample_size = K.cast(sample_size, dtype=K.dtype(normed_inputs))

            # sample variance - unbiased estimator of population variance
            batch_variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        # Update moving mean and moving variance
        self.moving_mean = K.moving_average_update(self.moving_mean,
                                                   batch_mean,
                                                   self.momentum)
        self.moving_variance = K.moving_average_update(self.moving_variance,
                                                       batch_variance,
                                                       self.momentum)
        # prepare broadcast shape
        normed_inputs = K.reshape(normed_inputs, group_shape)
        outputs = normed_inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(outputs,
                                lambda: normalize_inference(inputs),
                                training=training)

    def get_config(self):
        base_config = super(GroupNormalization, self).get_config()
        base_config.update(
            {
                'groups': self.groups,
                'axis': self.axis,
                'epsilon': self.epsilon,
                'center': self.center,
                'scale': self.scale,
                'beta_initializer': initializers.serialize(self.beta_initializer),
                'gamma_initializer': initializers.serialize(self.gamma_initializer),
                'beta_regularizer': regularizers.serialize(self.beta_regularizer),
                'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
                'beta_constraint': constraints.serialize(self.beta_constraint),
                'gamma_constraint': constraints.serialize(self.gamma_constraint)
            }
        )
        return base_config

    def compute_output_shape(self, input_shape):
        return input_shape
