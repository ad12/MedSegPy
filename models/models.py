import os, sys

from keras import Model
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D
from keras.utils import plot_model

sys.path.append('../')
import glob_constants as glc
from config import DeeplabV3Config, SegnetConfig, UNetConfig, \
    UNetMultiContrastConfig, UNet2_5DConfig, DeeplabV3_2_5DConfig, ResidualUNet, AnisotropicUNetConfig, RefineNetConfig, UNet3DConfig
from glob_constants import SEED

from models.deeplab_2d.deeplab_model import DeeplabModel
from models.segnet_2d.segnet import Segnet_v2
from models.unet_2d.residual_unet_model import residual_unet_2d
from models.unet_2d.anisotropic_unet_model import anisotropic_unet_2d
from models.unet_2d.unet_model import unet_2d_model, unet_2d_model_v2
from models.refinenet.refinenet_model import refinenet_model
from models.unet_3d_model import unet_3d_model


def get_model(config):
    """
    Get model corresponding to config
    :param config: Config object with specific architecture configurations
    :return: a Keras model
    """
    if (type(config) is DeeplabV3Config):
        model = deeplabv3_2d(config)
    elif (type(config) is SegnetConfig):
        model = segnet_2d(config)
    elif (type(config) is UNetConfig):
        model = unet_2d(config)
    elif (type(config) is UNetMultiContrastConfig):
        model = unet_2d_multi_contrast(config)
    elif (type(config) is UNet2_5DConfig):
        model = unet_2_5d(config)
    elif (type(config) is DeeplabV3_2_5DConfig):
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
        raise ValueError('This config type has not been implemented')

    # if weighted cross entropy, use softmax
    return model


def basic_refinenet(config):
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()
    model = refinenet_model(input_shape=input_shape)

    # Add activation
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
    model = Model(inputs=model.input, outputs=x)

    return model

def anisotropic_unet(config):
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()

    DEPTH = config.DEPTH
    NUM_FILTERS = config.NUM_FILTERS
    model = anisotropic_unet_2d(input_size=input_shape, depth=DEPTH, num_filters=NUM_FILTERS,
                                kernel_size=config.KERNEL_SIZE)

    # Add activation
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
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
    model = residual_unet_2d(input_size=input_shape, depth=DEPTH, num_filters=NUM_FILTERS,
                             layer_order=config.LAYER_ORDER,
                             dropout_rate=config.DROPOUT_RATE,
                             use_squeeze_excitation=config.USE_SE_BLOCK,
                             squeeze_excitation_ratio=config.SE_RATIO)

    # Add activation
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
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
    model = unet_3d_model(input_size=input_shape,
                          depth=DEPTH,
                          num_filters=NUM_FILTERS, num_classes=num_classes,
                          activation=activation)

    return model

def unet_2d(config):
    """
     Returns Unet2D model
     :param config: a UNetConfig object
     :return: a Keras model

     :raises ValueError: if config not of type UNetConfig
     """
    input_shape = config.IMG_SIZE
    activation = config.LOSS[1]
    num_classes = config.get_num_classes()

    # Legacy: some weights were trained on different structure (conv and activation were combined) making loading
    #           weights difficult.
    #           We check to see if we are testing and if we are in a case where we need to account for this issue
    if config.STATE == 'testing' and config.VERSION <= 2:
        model = unet_2d_model(input_size=input_shape, output_mode='sigmoid')
    else:
        DEPTH = config.DEPTH
        NUM_FILTERS = config.NUM_FILTERS
        model = unet_2d_model_v2(input_size=input_shape, depth=DEPTH, num_filters=NUM_FILTERS)

        # Add activation
        x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
        model = Model(inputs=model.input, outputs=x)

    return model


def deeplabv3_2d(config):
    """
    Returns DeeplabV3+ model
    :param config: a DeeplabV3Config object
    :return: a Keras model

    :raises ValueError: if config not of type DeeplabV3Config
    """
    if type(config) is not DeeplabV3Config:
        raise ValueError('config must be an instance of DeeplabV3Config')

    input_shape = config.IMG_SIZE
    OS = config.OS
    dil_rate_input = config.DIL_RATES
    activation = config.LOSS[1]
    dropout_rate = config.DROPOUT_RATE
    num_classes = config.get_num_classes()
    m = DeeplabModel(kernel_initializer=config.KERNEL_INITIALIZER, seed=glc.SEED)
    model = m.Deeplabv3(weights=None,
                        input_shape=input_shape,
                        classes=num_classes,
                        backbone='xception',
                        OS=OS,
                        dil_rate_input=dil_rate_input,
                        dropout_rate=dropout_rate)

    # Add sigmoid activation layer -
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
    model = Model(inputs=model.input, outputs=x)

    # Save image
    dil_rates_str = str(dil_rate_input[0]) + '-' + str(dil_rate_input[1]) + '-' + str(dil_rate_input[2])
    img_name = config.CP_SAVE_TAG + '_' + str(OS) + '_' + dil_rates_str + '.png'
    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, img_name), show_shapes=True)

    return model


def segnet_2d(config):
    """
    Returns SegnetConfig model
    :param config: a SegnetConfig object
    :return: a Keras model

    :raises ValueError: if config not of type SegnetConfig
    """
    if (type(config) is not SegnetConfig):
        raise ValueError('config must be an instance of SegnetConfig')
    num_classes = config.get_num_classes()
    input_shape = config.IMG_SIZE
    output_mode = config.LOSS[1]
    model = Segnet_v2(input_shape=input_shape,
                      n_labels=num_classes,
                      depth=config.DEPTH,
                      num_conv_layers=config.NUM_CONV_LAYERS,
                      num_filters=config.NUM_FILTERS,
                      single_bn=config.SINGLE_BN,
                      conv_act_bn=config.CONV_ACT_BN,
                      output_mode=output_mode)

    model_name = config.CP_SAVE_TAG + '_%d' + '_%s' + '_%s'
    bn_str = 'xbn'
    conv_act_bn_str = 'cba'

    if config.SINGLE_BN:
        bn_str = '1bn'

    if config.CONV_ACT_BN:
        conv_act_bn_str = 'cab'

    model_name = model_name % (config.DEPTH, bn_str, conv_act_bn_str)

    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, model_name + '.png'), show_shapes=True)

    return model


def unet_2d_multi_contrast(config):
    """
    Returns unet model corresponding to 3-channel multi-contrast inputs
    :param config: a UNetMultiContrastConfig object
    :return: a Keras model

    :raises ValueError: if config not of type UNetMultiContrastConfig
    """
    if (type(config) is not UNetMultiContrastConfig):
        raise ValueError('config must be instance of UNetMultiContrastConfig')

    activation = config.LOSS[1]
    num_classes = config.get_num_classes()
    input_shape = config.IMG_SIZE

    print('Initializing multi contrast 2d unet: input size - ' + str(input_shape))

    x = Input(input_shape)
    x = Conv2D(1, (1, 1), name='conv_mc_comp')(x)

    model = unet_2d_model(input_tensor=x)

    # Add activation
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
    model = Model(inputs=model.input, outputs=x)

    # only load weights for layers that share the same name
    if (config.INIT_UNET_2D):
        model.load_weights(config.INIT_UNET_2D_WEIGHTS, by_name=True)

    return model


def unet_2_5d(config):
    """
    Returns unet model corresponding to 3-channel multi-contrast inputs
    :param config: a UNetMultiContrastConfig object
    :return: a Keras model

    :raises ValueError: if config not of type UNetMultiContrastConfig
    """
    if (type(config) is not UNet2_5DConfig):
        raise ValueError('config must be instance of UNet2_5DConfig')

    activation = config.LOSS[1]
    num_classes = config.get_num_classes()
    input_shape = config.IMG_SIZE

    print('Initializing 2.5d unet: input size - ' + str(input_shape))

    x = Input(input_shape)

    model = unet_2d_model(input_tensor=x)

    # Add activation
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
    model = Model(inputs=model.input, outputs=x)

    # only load weights for layers that share the same name
    if (config.INIT_UNET_2D):
        model.load_weights(config.INIT_UNET_2D_WEIGHTS, by_name=True)

    return model


def deeplabv3_2_5d(config):
    """
    Returns unet model corresponding to 3-channel multi-contrast inputs
    :param config: a UNetMultiContrastConfig object
    :return: a Keras model

    :raises ValueError: if config not of type UNetMultiContrastConfig
    """
    if (type(config) is not DeeplabV3_2_5DConfig):
        raise ValueError('config must be instance of DeeplabV3_2_5DConfig')
    print('Initializing 2.5d deeplab: input size - ' + str(config.IMG_SIZE))

    input_shape = config.IMG_SIZE
    OS = config.OS
    dil_rate_input = config.DIL_RATES
    activation = config.LOSS[1]
    dropout_rate = config.DROPOUT_RATE
    num_classes = config.get_num_classes()
    m = DeeplabModel(kernel_initializer=config.KERNEL_INITIALIZER, seed=glc.SEED)
    model = m.Deeplabv3(weights=None,
                        input_shape=input_shape,
                        classes=num_classes,
                        backbone='xception',
                        OS=OS,
                        dil_rate_input=dil_rate_input,
                        dropout_rate=dropout_rate)

    # Add sigmoid activation layer -
    x = __add_activation_layer(output=model.layers[-1].output, num_classes=num_classes, activation=activation)
    model = Model(inputs=model.input, outputs=x)

    # Save image
    dil_rates_str = str(dil_rate_input[0]) + '-' + str(dil_rate_input[1]) + '-' + str(dil_rate_input[2])
    img_name = config.CP_SAVE_TAG + '_' + str(OS) + '_' + dil_rates_str + '.png'
    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, img_name), show_shapes=True)

    return model


def __softmax_activation_layer(output, num_classes):
    """
    Returns softmax activation layer
    :param output:
    :param num_classes:
    :return:
    """
    return


def __add_activation_layer(output, num_classes, activation='sigmoid'):
    """
    Return sigmoid activation layer
    :param: output: The output of the previous layer
    """

    # Initializing kernel weights to 1 and bias to 0.
    # i.e. without training, the output would be a sigmoid activation on each pixel of the input
    return Conv2D(num_classes, (1, 1), activation=activation,
                  kernel_initializer=glorot_uniform(seed=SEED),
                  name='output_activation')(output)


if __name__ == '__main__':
    # config = UNetMultiContrastConfig(create_dirs=False)
    # config.INIT_UNET_2D_WEIGHTS = './test_data/unet_2d_fc_weights.004--0.8968.h5'
    #
    # unet_2d_multi_contrast(config)
    config = RefineNetConfig(create_dirs=False)
    save_path = '../imgs/refinenet_resnet50.png'
    m = get_model(config)
    plot_model(m, to_file=save_path, show_shapes=True)
