import logging
import tensorflow as tf
import warnings

from .build import META_ARCH_REGISTRY, ModelBuilder
from keras import Input
from keras.engine import get_source_inputs
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import (
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    MaxPooling2D
)
from medsegpy import config
from medsegpy.config import (
    ContextEncoderConfig,
    ContextUNetConfig,
    ContextInpaintingConfig,
    ContextSegmentationConfig
)
from medsegpy.modeling.layers.normalization import GroupNormalization as GroupNorm
from medsegpy.modeling.layers.convolutional import ConvStandardized2D
from medsegpy.modeling.load_weights_utils import (
    SelfSupervisedInfo
)
from medsegpy.modeling.model_utils import (
    zero_pad_like,
    add_sem_seg_activation
)
from ..model import Model
from typing import Dict, Sequence, Tuple, Union

logger = logging.getLogger(__name__)


@META_ARCH_REGISTRY.register()
class ContextEncoder(ModelBuilder):
    def __init__(self, cfg: ContextEncoderConfig):
        """
        "Context Encoder" is the name of an encoder network defined in the paper
        "Context Encoders: Feature Learning by Inpainting" (Pathak et al.), which
        learns to create a compact representation of an image that can then be
        used by a decoder network to fill in missing regions of the input image,
        which is called "inpainting".
        For this class, the context encoder architecture will be the same
        as the encoder for a standard 2D UNet. This is different from the original
        paper, but the actual encoder architecture should not matter too much.
        Args:
            cfg: the configuration file for the Context Encoder
        """
        super().__init__(cfg)
        self._cfg = cfg
        self._conv_type = Conv2D
        self._pooler_type = MaxPooling2D
        self._dim = 2
        self._kernel_size = (3, 3)

        # Get type of convolution
        self.weight_standardization = self._cfg.WEIGHT_STANDARDIZATION

        # Get type of layer normalization
        layer_norm_type = self._cfg.NORMALIZATION
        self._layer_norm = None

        if layer_norm_type == "NoNorm":
            self._layer_norm = None
        elif layer_norm_type == "BatchNorm":
            self._layer_norm = BatchNorm
        elif layer_norm_type == "GroupNorm":
            self._layer_norm = GroupNorm
        else:
            warnings.warn(f"The normalization type \"{layer_norm_type}\" "
                          f"is not supported. Batch Norm will be used.")
            self._layer_norm = BatchNorm
        self._layer_norm_args = self._cfg.NORMALIZATION_ARGS

        # Store number of specific layer types for naming each layer
        self._num_conv = 0
        self._num_bn = 0
        self._num_dropout = 0
        self._num_pool = 0

        # Store pool sizes for decoder network
        self.pool_sizes = []

    def build_model(self, input_tensor=None) -> Model:
        """
        Builds the encoder network and returns the resulting model.
        This implementation will overload the abstract method defined in the
        superclass "ModelBuilder".
        Args:
            input_tensor: The input to the network.
        Returns:
            model: A Model that defines the encoder network.
        """
        # Get architecture hyperparameters
        cfg = self._cfg
        seed = cfg.SEED
        kernel_size = self._kernel_size
        kernel_initializer = {
            "class_name": cfg.KERNEL_INITIALIZER,
            "config": {"seed": seed},
        }
        num_filters = cfg.NUM_FILTERS
        depth = len(num_filters)

        # Build inputs
        if input_tensor is None:
            inputs = self._build_input(cfg.IMG_SIZE)
        else:
            inputs = input_tensor

        # Encoder
        x = inputs
        x_skips = []
        for depth_cnt in range(depth):
            x = self.build_encoder_block(
                x,
                num_filters[depth_cnt],
                kernel_size=kernel_size,
                activation="relu",
                kernel_initializer=kernel_initializer,
                dropout=0.0,
            )

            # Maxpool until penultimate depth.
            if depth_cnt < depth - 1:
                x_skips.append(x)
                pool_size = self._get_pool_size(x)
                self.pool_sizes.append(pool_size)
                self._num_pool += 1
                x = self._pooler_type(pool_size=pool_size,
                                      name="encoder_pool_%d" % self._num_pool
                                      )(x)

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        model = Model(inputs=inputs, outputs=[x, *x_skips])

        return model

    def build_encoder_block(
            self,
            x: tf.Tensor,
            num_filters: Sequence[int],
            kernel_size: Union[int, Sequence[int]] = 3,
            activation: str = "relu",
            kernel_initializer: Union[str, Dict] = "he_normal",
            dropout: float = 0.0,
    ) -> tf.Tensor:
        """
        Builds one block of the ContextEncoder.
        Each block consists of the following structure:
        [Conv -> Activation] -> BN -> Dropout.
        Args:
            x: Input tensor.
            num_filters: Number of filters to use for each conv layer.
            kernel_size: Kernel size accepted by Keras convolution layers.
            activation: Activation type.
            kernel_initializer: Kernel initializer accepted by
                                `keras.layers.Conv(...)`.
            dropout: Dropout rate.
        Returns:
            Output of encoder block.
        """
        num_conv_layers = len(num_filters)
        for i, filters in enumerate(num_filters):
            self._num_conv += 1
            if self.weight_standardization and i == num_conv_layers - 1:
                x = ConvStandardized2D(
                    filters,
                    kernel_size,
                    padding="same",
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    name="encoder_conv_%d" % self._num_conv
                )(x)
            else:
                x = self._conv_type(
                    filters,
                    kernel_size,
                    padding="same",
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    name="encoder_conv_%d" % self._num_conv
                )(x)

        if self._layer_norm is not None:
            self._num_bn += 1
            norm_name = "encoder_bn_%d" % self._num_bn
            x = self._layer_norm(
                **self._layer_norm_args,
                name=norm_name
            )(x, **SelfSupervisedInfo.set_to_inference(layer_name=norm_name))

        self._num_dropout += 1
        dropout_name = "encoder_dropout_%d" % self._num_dropout
        x = Dropout(
            rate=dropout,
            name=dropout_name
        )(x)

        return x

    @staticmethod
    def _build_input(input_size: Tuple):
        """
        Creates an input tensor of size "input_size".
        Args:
            input_size: The size of the input tensor for the model.
        Returns:
            A symbolic input (i.e. a placeholder) with size "input_size".
        """
        if len(input_size) == 2:
            input_size = input_size + (1,)
        elif len(input_size) != 3:
            raise ValueError("2D ContextEncoder must have an input of shape HxWxC")
        return Input(input_size)

    @staticmethod
    def _get_pool_size(x: tf.Tensor):
        """
        Determines the right size of the pooling filter based on the
        dimensions of the input tensor `x`.
        Args:
            x: The input tensor.
        Returns:
            A list with the same number of elements as dimensions of `x`,
            where each element is either 2 or 3.
        """
        return [
            2 if d % 2 == 0 or d % 3 != 0 else 3
            for d in x.shape.as_list()[1:-1]
        ]


@META_ARCH_REGISTRY.register()
class ContextUNet(ModelBuilder):
    def __init__(self, cfg: ContextUNetConfig):
        """
        This model implements a basic U-Net structure that takes an input
        image, encodes it with an instance of the ContextEncoder model, and
        the decodes the output of the encoder until the decoded output has
        the same resolution as the input image. The model will then end
        there, without any post-processing.
        For this class, the architecture will the same as a standard 2D UNet.
        The encoder will be the ContextEncoder model and the decoder will
        consist of the appropriate type and number of layers to output a
        feature map with the same size as the original input, given the output
        from the encoder model.
        """
        super().__init__(cfg)
        self._cfg = cfg
        self._conv_type = Conv2D
        self._conv_transpose_type = Conv2DTranspose
        self._dim = 2
        self._kernel_size = (3, 3)

        # Get type of convolution
        self.weight_standardization = self._cfg.WEIGHT_STANDARDIZATION

        # Get type of layer normalization
        layer_norm_type = self._cfg.NORMALIZATION
        self._layer_norm = None

        if layer_norm_type == "NoNorm":
            self._layer_norm = None
        elif layer_norm_type == "BatchNorm":
            self._layer_norm = BatchNorm
        elif layer_norm_type == "GroupNorm":
            self._layer_norm = GroupNorm
        else:
            warnings.warn(f"The normalization type \"{layer_norm_type}\" "
                          f"is not supported. Batch Norm will be used.")
            self._layer_norm = BatchNorm
        self._layer_norm_args = self._cfg.NORMALIZATION_ARGS

        self._kernel_initializer = {
            "class_name": cfg.KERNEL_INITIALIZER,
            "config": {"seed": cfg.SEED},
        }
        self._decoder_suffix = ""

        # Store number of specific layer types for naming each layer
        self._num_conv = 0
        self._num_conv_t = 0
        self._num_norm = 0
        self._num_dropout = 0

    def build_model(self, input_tensor=None) -> Model:
        """
        Builds the full encoder/decoder architecture and returns the
        resulting model.
        This implementation will overload the abstract method defined in the
        superclass "ModelBuilder".
        Args:
            input_tensor: The input to the network.
        Returns:
            model: A Model that defines the full encoder/decoder architecture.
        """
        cfg = self._cfg

        # Get architecture hyperparameters
        num_filters = cfg.NUM_FILTERS
        kernel_size = self._kernel_size

        # Build inputs
        if input_tensor is None:
            inputs = self._build_input(cfg.IMG_SIZE)
        else:
            inputs = input_tensor

        # Read ContextEncoder config
        context_encoder_config = config.get_config("ContextEncoder",
                                                   create_dirs=False)
        # Overwrite certain config settings
        context_encoder_config.NUM_FILTERS = num_filters
        context_encoder_config.NORMALIZATION = cfg.NORMALIZATION
        context_encoder_config.NORMALIZATION_ARGS = cfg.NORMALIZATION_ARGS
        context_encoder_config.WEIGHT_STANDARDIZATION = cfg.WEIGHT_STANDARDIZATION
        context_encoder_config.SEED = cfg.SEED
        context_encoder_config.KERNEL_INITIALIZER = cfg.KERNEL_INITIALIZER
        context_encoder_config.IMG_SIZE = cfg.IMG_SIZE

        # depth = length of NUM_FILTERS list
        depth = len(num_filters)

        # Build ContextEncoder model
        encoder_name_in_model = "context_encoder_model"
        context_encoder_builder = ContextEncoder(context_encoder_config)
        context_encoder = context_encoder_builder.build_model(inputs)
        context_encoder.name = encoder_name_in_model
        pool_sizes = context_encoder_builder.pool_sizes
        
        # Pass input through ContextEncoder
        x = inputs

        encoder_outputs = context_encoder(x)
        x = encoder_outputs[0]
        x_skips = encoder_outputs[1:]

        # Decoder
        for i, (x_skip, unpool_size) in enumerate(
                zip(x_skips[::-1], pool_sizes[::-1])
        ):
            depth_cnt = depth - i - 2
            x = self.build_decoder_block(
                x,
                x_skip,
                num_filters[depth_cnt],
                unpool_size,
                kernel_size=kernel_size,
                activation="relu",
                kernel_initializer=self._kernel_initializer,
                dropout=0.0,
            )

        x = self.post_processing(x)

        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)

        model = Model(inputs=inputs, outputs=[x])

        return model

    def post_processing(self, x: tf.Tensor):
        """
        Post-processes the output of ContextUnet based on the user's
        preferences.
        The base ContextUnet will not post-process the output. Other models
        that inherit ContextUnet can overload this function to post-process
        the output of the base ContextUnet.
        Args:
            x: Output of base ContextUnet.
        Return:
            Post-processed output.
        """
        return x

    def build_decoder_block(
            self,
            x: tf.Tensor,
            x_skip: tf.Tensor,
            num_filters: Sequence[int],
            unpool_size: Sequence[int],
            kernel_size: Union[int, Sequence[int]] = 3,
            activation: str = "relu",
            kernel_initializer: Union[str, Dict] = "he_normal",
            dropout: float = 0.0
    ):
        """
        Builds one block of the decoder.
        Each block of the decoder will have the following structure:
        Input -> Transposed Convolution -> Concatenate Skip Connection
        -> Convolution -> BN -> Dropout (optional)
        Args:
            x: Input tensor.
            x_skip: The output of the next highest layer of the
                    ContextEncoder.
            num_filters: Number of filters for both the transposed convolution
                            and regular convolution.
            unpool_size: The stride for the transposed convolutional
                            layer in order to increase the resolution of the
                            input to that of the next highest layer in the
                            ContextEncoder.
            kernel_size: The filter size for all transposed convolutional and
                            regular convolutional layers.
            activation: The activation to use for all transposed convolutional
                            and regular convolution layers.
            kernel_initializer: The kernel initializer for all transposed
                                    convolutional and regular convolutional
                                    layers.
            dropout: Dropout rate.
        Returns:
           Output of the decoder block.
        """
        x_shape = x.shape.as_list()[1:-1]
        x = self._conv_transpose_type(
            num_filters[0],
            kernel_size,
            padding="same",
            strides=unpool_size,
            kernel_initializer=kernel_initializer,
            name=f"conv2d_transpose_{self._num_conv_t}_{self._decoder_suffix}"
        )(x)
        self._num_conv_t += 1
        x_shape = [x * y for x, y in zip(x_shape, unpool_size)]
        x = Concatenate(axis=-1)([zero_pad_like(x, x_skip, x_shape), x_skip])

        if len(num_filters) > 1:
            num_conv_layers = len(num_filters[1:])
            for i, filters in enumerate(num_filters[1:]):
                if self.weight_standardization and i == num_conv_layers - 1:
                    x = ConvStandardized2D(
                        filters,
                        kernel_size,
                        padding="same",
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        name=f"conv_standardized2d_{self._num_conv}_"
                             f"{self._decoder_suffix}"
                    )(x)
                else:
                    x = self._conv_type(
                        filters,
                        kernel_size,
                        padding="same",
                        activation=activation,
                        kernel_initializer=kernel_initializer,
                        name=f"conv2d_{self._num_conv}_{self._decoder_suffix}"
                    )(x)
                self._num_conv += 1
            if self._layer_norm is not None:
                norm_name = f"normalization_{self._num_norm}_" \
                            f"{self._decoder_suffix}"
                x = self._layer_norm(
                    **self._layer_norm_args,
                    name=norm_name
                )(x,
                  **SelfSupervisedInfo.set_to_inference(layer_name=norm_name))
                self._num_norm += 1
            dropout_name = f"dropout_{self._num_dropout}_{self._decoder_suffix}"
            x = Dropout(
                rate=dropout,
                name=dropout_name
            )(x)
            self._num_dropout += 1
        return x

    @staticmethod
    def _build_input(input_size: Tuple):
        """
        Creates an input tensor of size "input_size".
        Args:
            input_size: The size of the input tensor for the model.
        Returns:
            A symbolic input (i.e. a placeholder) with size "input_size".
        """
        if len(input_size) == 2:
            input_size = input_size + (1,)
        elif len(input_size) != 3:
            raise ValueError("2D ContextInpainting must have an input of shape HxWxC")
        return Input(input_size)


@META_ARCH_REGISTRY.register()
class ContextInpainting(ContextUNet):
    def __init__(self, cfg: ContextInpaintingConfig):
        """
        The ContextInpainting model builds on top of the base ContextUNet model.
        This model receives an input image with some regions cut out and
        filled in with some value, and tries to reconstruct the original
        image given this input. This task was defined in the paper:
        "Context Encoders: Feature Learning by Inpainting" (Pathak et al.).
        The ContextInpainting model will pass the image through the base
        ContextUNet model, and then post-process the output of ContextUNet
        to create a 1-channel output image, which is the model's prediction
        of what it believes to be the original image.
        This model will be used to pre-train the ContextEncoder model, such
        that the compact representations outputted by the ContextEncoder will
        have some useful information about the input dataset. The pretrained
        ContextEncoder model will then be used for the downstream task of
        image segmentation.
        """
        super().__init__(cfg)
        self._cfg = cfg
        self._decoder_suffix = "inpainting_decoder"

    def post_processing(self, x: tf.Tensor):
        # Final convolutional layer to get to cfg.IMG_SIZE[-1] channel output
        x = self._conv_type(
            self._cfg.IMG_SIZE[-1],
            kernel_size=self._kernel_size,
            padding="same",
            kernel_initializer=self._kernel_initializer,
            name="post_process"
        )(x)
        return x


@META_ARCH_REGISTRY.register()
class ContextSegmentation(ContextUNet):
    """
    The ContextSegmentation model will build off the base ContextUNet model,
    just like the ContextInpainting model. However, for this model, the
    output of ContextUNet will be post-processed such that the model
    outputs segmentation probabilities for each segmentation class.
    The ContextSegmentation model will use a ContextEncoder model whose
    weights are pretrained based on the inpainting task.
    For the first pass of training, the ContextEncoder model will be frozen
    and set to inference mode, such that the batchnorm statistics after
    pretraining are not ruined. The rest of the weights for the
    ContextSegmentation model will be trained using
    the training dataset.
    The second pass of training will unfreeze the ContextEncoder model,
    keeping the ContextEncoder in inference mode, and will fine-tune the
    trainable weights of the ContextEncoder model.
    """
    def __init__(self, cfg: ContextSegmentationConfig):
        super().__init__(cfg)
        self._cfg = cfg
        self._decoder_suffix = "segmentation_decoder"

    def post_processing(self, x: tf.Tensor):
        # Returns probabilities for each segmentation class
        cfg = self._cfg
        num_classes = cfg.get_num_classes()
        x = add_sem_seg_activation(
            x,
            num_classes=num_classes,
            conv_type=self._conv_type,
            kernel_initializer=self._kernel_initializer,
            layer_name="post_process"
        )
        return x
