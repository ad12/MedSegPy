from keras.utils import plot_model
from keras.layers import Input, Conv2D, Concatenate
from keras import Model
from keras.initializers import Zeros, Ones, Constant

from config import DeeplabV3Config, SegnetConfig, UNetConfig, \
                    EnsembleUDSConfig, UNetMultiContrastConfig, UNet2_5DConfig, DeeplabV3_2_5DConfig
from deeplab_2d.deeplab_model import Deeplabv3
from segnet_2d.segnet import Segnet, Segnet_v2
from unet_2d.unet_model import unet_2d_model

import os


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
    elif (type(config) is EnsembleUDSConfig):
        model = ensemble_uds(config)
    elif (type(config) is UNetMultiContrastConfig):
        model = unet_2d_multi_contrast(config)
    elif (type(config) is UNet2_5DConfig):
        model = unet_2_5d(config)
    elif (type(config) is DeeplabV3_2_5DConfig):
        model = deeplabv3_2_5d(config)
    else:
        raise ValueError('This config type has not been implemented')

    # if weighted cross entropy, use softmax
    return model

def unet_2d(config):
    """
     Returns Unet2D model
     :param config: a UNetConfig object
     :return: a Keras model

     :raises ValueError: if config not of type UNetConfig
     """
    input_shape = config.IMG_SIZE
    output_mode = config.LOSS[1]
    model = unet_2d_model(input_size=input_shape, output_mode=output_mode)

    return model

def deeplabv3_2d(config):
    """
    Returns DeeplabV3+ model
    :param config: a DeeplabV3Config object
    :return: a Keras model

    :raises ValueError: if config not of type DeeplabV3Config
    """
    if (type(config) is not DeeplabV3Config):
        raise ValueError('config must be an instance of DeeplabV3Config')

    input_shape = config.IMG_SIZE
    OS = config.OS
    dil_rate_input = config.DIL_RATES
    activation = config.LOSS[1]
    dropout_rate = config.DROPOUT_RATE
    num_classes = config.get_num_classes()
    model = Deeplabv3(weights=None,
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
        conv_act_bn_str='cab'

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
    print('Initializing multi contrast 2d unet: input size - ' + str(config.IMG_SIZE))
    input_shape = config.IMG_SIZE
    x = Input(input_shape)
    x = Conv2D(1, (1,1), name='conv_mc_comp')(x)
    model = unet_2d_model(input_tensor=x)

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
    print('Initializing 2.5d unet: input size - ' + str(config.IMG_SIZE))
    input_shape = config.IMG_SIZE
    x = Input(input_shape)
    model = unet_2d_model(input_tensor=x)

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
    model = Deeplabv3(weights=None,
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

def ensemble_uds(config):
    """
    Returns model corresponding to ensemble of unet, deeplab, and segnet
    :param config: a EnsembleUDSConfig object
    :return: a Keras model

    :raises ValueError: if config not of type EnsembleUDSConfig
    """
    if (type(config) is not EnsembleUDSConfig):
        raise ValueError('config must be instance of EnsembleUDSConfig')
    # Deeplab
    input_shape = config.IMG_SIZE

    num_classes = config.get_num_classes()
    deeplab_model = Deeplabv3(weights=None, input_shape=input_shape, classes=num_classes, backbone='xception', OS=config.OS, dil_rate_input=config.DIL_RATES)
    deeplab_model.load_weights(config.DEEPLAB_INIT_WEIGHTS, by_name=True)
    deeplab_model.trainable = False
    x = deeplab_model.input

    # Unet
    unet_model = unet_2d_model(input_size=input_shape, input_tensor=x)
    unet_model.load_weights(config.UNET_INIT_WEIGHTS, by_name=True)
    unet_model.trainable = False

    # Segnet
    segnet_model = Segnet(input_tensor=x, n_labels=num_classes)
    print('Loaded Segnet')
    segnet_model.load_weights(config.SEGNET_INIT_WEIGHTS)
    segnet_model.trainable = False

    model = combine_models(x, [unet_model, deeplab_model, segnet_model], ensemble_name='ensemble_uds')
    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, config.CP_SAVE_TAG + '.png'), show_shapes = True)

    return model


def combine_models(x_input, models, ensemble_name='ensemble', num_classes=1):
    """
    Combine Keras models to form single ensemble model
    :param x_input: a Keras Input tensor
    :param models: list of Keras models
    :param ensemble_name: name of ensemble model
    :param num_classes: number of output classes
    :return:
    """
    outputs = []
    for model in models:
        outputs.append(model.layers[-1].output)
    
    x = Concatenate(name='%s_conc' % ensemble_name)(outputs)
    x = Conv2D(num_classes, (1,1), name = '%s_conv' % ensemble_name, activation='sigmoid', kernel_initializer=Constant(value=1.0/len(models)))(x)

    model = Model(inputs=x_input, outputs=x)
    for layer in model.layers[:-1]:
        layer.trainable = False
    
    model.summary()
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
    return Conv2D(num_classes, (1,1), activation=activation, kernel_initializer=Ones(), bias_initializer=Zeros(), name='output_activation')(output)


if __name__ == '__main__':
    config = UNetMultiContrastConfig(create_dirs=False)
    config.INIT_UNET_2D_WEIGHTS = './test_data/unet_2d_fc_weights.004--0.8968.h5'

    unet_2d_multi_contrast(config)
