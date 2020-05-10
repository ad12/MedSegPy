import logging
import warnings

from keras.initializers import glorot_uniform
from keras.layers import Conv2D, Input

from medsegpy.config import (
    AnisotropicUNetConfig,
    DeeplabV3_2_5DConfig,
    DeeplabV3Config,
    RefineNetConfig,
    ResidualUNet,
    SegnetConfig,
    UNet2_5DConfig,
    UNet3DConfig,
    UNetConfig,
)

from .deeplab_2d.deeplab_model import DeeplabModel
from .model import Model
from .refinenet.refinenet_model import refinenet_model
from .segnet_2d.segnet import Segnet_v2
from .segnet_2d.segnet_bottleneck import SegNetBottleneck
from .unet_2d.anisotropic_unet_model import anisotropic_unet_2d
from .unet_2d.residual_unet_model import residual_unet_2d
from .unet_2d.unet_model import unet_2d_model, unet_2d_model_v2
from .unet_3d_model import unet_3d_model
from .meta_arch.build import build_model

logger = logging.getLogger(__name__)


def get_model(config):
    """
    Get model corresponding to config
    :param config: Config object with specific architecture configurations
    :return: a Keras model
    """
    if type(config) is DeeplabV3Config:
        model = deeplabv3_2d(config)
    elif type(config) is SegnetConfig:
        model = segnet_2d(config)
    elif type(config) is UNetConfig:
        #model = unet_2d(config)
        model = build_model(config)
    elif type(config) is UNet2_5DConfig:
        model = unet_2_5d(config)
    elif type(config) is DeeplabV3_2_5DConfig:
        model = deeplabv3_2_5d(config)
    elif type(config) is ResidualUNet:
        model = residual_unet(config)
    elif type(config) is AnisotropicUNetConfig:
        model = anisotropic_unet(config)
    elif type(config) is RefineNetConfig:
        model = basic_refinenet(config)
    elif type(config) is UNet3DConfig:
        model = unet_3d(config)
    else:
        raise ValueError("This config type has not been implemented")

    # if weighted cross entropy, use softmax
    return model


def basic_refinenet(config):
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()
    model = refinenet_model(input_shape=input_shape)

    # Add activation
    x = __add_activation_layer(
        output=model.layers[-1].output,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )
    model = Model(inputs=model.input, outputs=x)

    return model


def anisotropic_unet(config):
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()

    DEPTH = config.DEPTH
    NUM_FILTERS = config.NUM_FILTERS
    model = anisotropic_unet_2d(
        input_size=input_shape,
        depth=DEPTH,
        num_filters=NUM_FILTERS,
        kernel_size=config.KERNEL_SIZE,
    )

    # Add activation
    x = __add_activation_layer(
        output=model.layers[-1].output,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )
    model = Model(inputs=model.input, outputs=x)

    return model


def residual_unet(config):
    """
    Returns ResidualUnet model
    :param config:
    :return:
    """
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()

    DEPTH = config.DEPTH
    NUM_FILTERS = config.NUM_FILTERS
    model = residual_unet_2d(
        input_size=input_shape,
        depth=DEPTH,
        num_filters=NUM_FILTERS,
        layer_order=config.LAYER_ORDER,
        dropout_rate=config.DROPOUT_RATE,
        use_squeeze_excitation=config.USE_SE_BLOCK,
        squeeze_excitation_ratio=config.SE_RATIO,
    )

    # Add activation
    x = __add_activation_layer(
        output=model.layers[-1].output,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )
    model = Model(inputs=model.input, outputs=x)

    return model


def unet_3d(config):
    """
     Returns Unet3D model
     :param config: a UNetConfig object
     :return: a Keras model

     :raises ValueError: if config not of type UNetConfig
     """
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()

    DEPTH = config.DEPTH
    NUM_FILTERS = config.NUM_FILTERS
    model = unet_3d_model(
        input_size=input_shape,
        depth=DEPTH,
        num_filters=NUM_FILTERS,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )

    return model


def unet_2d(config):
    """
     Returns Unet2D model
     :param config: a UNetConfig object
     :return: a Keras model

     :raises ValueError: if config not of type UNetConfig
     """
    warnings.warn(
        "unet_2d is deprecated. Use `meta_arch.build_model(...)`.",
        DeprecationWarning,
    )
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()

    # Legacy: some weights were trained on different structure
    # (conv and activation were combined) making loading weights difficult.
    # We check to see if we are testing and if we are in a case where we need
    # to account for this issue.
    if config.STATE == "testing" and config.VERSION <= 2:
        model = unet_2d_model(input_size=input_shape, output_mode="sigmoid")
    else:
        DEPTH = config.DEPTH
        NUM_FILTERS = config.NUM_FILTERS
        model = unet_2d_model_v2(
            input_size=input_shape, depth=DEPTH, num_filters=NUM_FILTERS
        )

        # Add activation
        x = __add_activation_layer(
            output=model.layers[-1].output,
            num_classes=num_classes,
            activation=activation,
            seed=config.SEED,
        )
        model = Model(inputs=model.input, outputs=x)

    return model


def deeplabv3_2d(config):
    """
    Returns DeeplabV3+ model
    :param config: a DeeplabV3Config object
    :return: a Keras model

    :raises ValueError: if config not of type DeeplabV3Config
    """
    warnings.warn(
        "deeplabv3_2d is deprecated. Use `meta_arch.build_model(...)`.",
        DeprecationWarning,
    )

    if type(config) is not DeeplabV3Config:
        raise ValueError("config must be an instance of DeeplabV3Config")

    input_shape = config.IMG_SIZE
    OS = config.OS
    dil_rate_input = config.DIL_RATES
    activation = config.LOSS[1]
    dropout_rate = config.DROPOUT_RATE
    num_classes = config.get_num_classes()
    m = DeeplabModel(
        kernel_initializer=config.KERNEL_INITIALIZER, seed=config.SEED
    )
    model = m.Deeplabv3(
        weights=None,
        input_shape=input_shape,
        classes=num_classes,
        backbone="xception",
        OS=OS,
        dil_rate_input=dil_rate_input,
        dropout_rate=dropout_rate,
    )

    # Add sigmoid activation layer -
    x = __add_activation_layer(
        output=model.layers[-1].output,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )
    model = Model(inputs=model.input, outputs=x)

    return model


def segnet_2d(config: SegnetConfig):
    """
    Returns SegnetConfig model
    :param config: a SegnetConfig object
    :return: a Keras model

    :raises ValueError: if config not of type SegnetConfig
    """
    if type(config) is not SegnetConfig:
        raise ValueError("config must be an instance of SegnetConfig")
    num_classes = config.get_num_classes()
    input_shape = config.IMG_SIZE
    output_mode = config.LOSS[1]
    if config.USE_BOTTLENECK:
        model = SegNetBottleneck(
            input_shape=input_shape,
            n_labels=num_classes,
            depth=config.DEPTH,
            num_conv_layers=config.NUM_CONV_LAYERS,
            num_filters=config.NUM_FILTERS,
            single_bn=config.SINGLE_BN,
            conv_act_bn=config.CONV_ACT_BN,
            output_mode=output_mode,
            seed=config.SEED,
        )
        model = model.build_model()
    else:
        model = Segnet_v2(
            input_shape=input_shape,
            n_labels=num_classes,
            depth=config.DEPTH,
            num_conv_layers=config.NUM_CONV_LAYERS,
            num_filters=config.NUM_FILTERS,
            single_bn=config.SINGLE_BN,
            conv_act_bn=config.CONV_ACT_BN,
            output_mode=output_mode,
            seed=config.SEED,
        )

    return model


def unet_2_5d(config):
    """
    Returns unet model corresponding to 3-channel multi-contrast inputs
    :param config: a UNetMultiContrastConfig object
    :return: a Keras model

    :raises ValueError: if config not of type UNetMultiContrastConfig
    """
    warnings.warn(
        "unet_2_5d is deprecated. Use `meta_arch.build_model(...)`.",
        DeprecationWarning,
    )
    if type(config) is not UNet2_5DConfig:
        raise ValueError("config must be instance of UNet2_5DConfig")

    activation = config.LOSS[1]
    num_classes = config.get_num_classes()
    input_shape = config.IMG_SIZE

    logger.info("Initializing 2.5d unet: input size - " + str(input_shape))

    x = Input(input_shape)

    model = unet_2d_model_v2(input_tensor=x)

    # Add activation
    x = __add_activation_layer(
        output=model.layers[-1].output,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )
    model = Model(inputs=model.input, outputs=x)

    # only load weights for layers that share the same name
    if config.INIT_UNET_2D:
        model.load_weights(config.INIT_UNET_2D_WEIGHTS, by_name=True)

    return model


def deeplabv3_2_5d(config):
    """
    Returns unet model corresponding to 3-channel multi-contrast inputs
    :param config: a UNetMultiContrastConfig object
    :return: a Keras model

    :raises ValueError: if config not of type UNetMultiContrastConfig
    """
    warnings.warn(
        "deeplabv3_2_5d is deprecated. Use `meta_arch.build_model(...)`.",
        DeprecationWarning,
    )
    if type(config) is not DeeplabV3_2_5DConfig:
        raise ValueError("config must be instance of DeeplabV3_2_5DConfig")
    logger.info(
        "Initializing 2.5d deeplab: input size - " + str(config.IMG_SIZE)
    )

    input_shape = config.IMG_SIZE
    OS = config.OS
    dil_rate_input = config.DIL_RATES
    activation = config.LOSS[1]
    dropout_rate = config.DROPOUT_RATE
    num_classes = config.get_num_classes()
    m = DeeplabModel(
        kernel_initializer=config.KERNEL_INITIALIZER, seed=config.SEED
    )
    model = m.Deeplabv3(
        weights=None,
        input_shape=input_shape,
        classes=num_classes,
        backbone="xception",
        OS=OS,
        dil_rate_input=dil_rate_input,
        dropout_rate=dropout_rate,
    )

    # Add sigmoid activation layer -
    x = __add_activation_layer(
        output=model.layers[-1].output,
        num_classes=num_classes,
        activation=activation,
        seed=config.SEED,
    )
    model = Model(inputs=model.input, outputs=x)

    return model


def __add_activation_layer(
    output, num_classes, activation="sigmoid", seed=None
):
    """
    Return sigmoid activation layer
    :param: output: The output of the previous layer
    """

    # Initializing kernel weights to 1 and bias to 0.
    # i.e. without training, the output would be a sigmoid activation on each
    # pixel of the input
    return Conv2D(
        num_classes,
        (1, 1),
        activation=activation,
        kernel_initializer=glorot_uniform(seed=seed),
        name="output_activation",
    )(output)
