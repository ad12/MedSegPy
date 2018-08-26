from keras.utils import plot_model
from keras.layers import Input, Conv2D, Concatenate, ZeroPadding2D, Cropping2D
from keras import Model
from keras.initializers import Zeros, Ones, Constant

from config import DeeplabV3Config, SegnetConfig, UNetConfig, EnsembleUDSConfig, UNetMultiContrastConfig
from deeplab_2d.deeplab_model import Deeplabv3
from segnet_2d.segnet import Segnet, Segnet_v2
from unet_2d.unet_model import unet_2d_model

import os


def get_model(config):
    if (type(config) is DeeplabV3Config):
        return deeplabv3_2d(config)
    elif (type(config) is SegnetConfig):
        return segnet_2d(config)
    elif (type(config) is UNetConfig):
        return unet_2d(config)
    elif (type(config) is EnsembleUDSConfig):
        return ensemble_uds(config)
    elif (type(config) is UNetMultiContrastConfig):
        return unet_2d_multi_contrast(config)
    else:
        raise ValueError('This config type has not been implemented') 


def unet_2d_multi_contrast(config):
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

def ensemble_uds(config):
    if (type(config) is not EnsembleUDSConfig):
        raise ValueError('config must be instance of EnsembleUDSConfig')
    # Deeplab
    input_shape = config.IMG_SIZE

    deeplab_model = Deeplabv3(weights=None, input_shape=input_shape, classes=config.NUM_CLASSES, backbone='xception', OS=config.OS, dil_rate_input=config.DIL_RATES)
    deeplab_model.load_weights(config.DEEPLAB_INIT_WEIGHTS, by_name=True)
    deeplab_model.trainable = False
    x = deeplab_model.input

    # Unet
    unet_model = unet_2d_model(input_size=input_shape, input_tensor=x)
    unet_model.load_weights(config.UNET_INIT_WEIGHTS, by_name=True)
    unet_model.trainable = False

    # Segnet
    segnet_model = Segnet(input_tensor=x, n_labels=config.NUM_CLASSES)
    print('Loaded Segnet')
    segnet_model.load_weights(config.SEGNET_INIT_WEIGHTS)
    segnet_model.trainable = False

    model = combine_models(x, [unet_model, deeplab_model, segnet_model], ensemble_name='ensemble_uds')
    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, config.CP_SAVE_TAG + '.png'), show_shapes = True)

    return model

def combine_models(x_input, models, ensemble_name='ensemble', num_classes=1):
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


def unet_2d(config):
    input_shape = config.IMG_SIZE
    model = unet_2d_model(input_size=input_shape)

    return model

def deeplabv3_2d(config):
    if (type(config) is not DeeplabV3Config):
        raise ValueError('config must be an instance of DeeplabConfig')

    input_shape = config.IMG_SIZE
    OS = 16

    model = Deeplabv3(weights=None, input_shape=input_shape, classes=config.NUM_CLASSES, backbone='xception', OS=OS, dilation_divisor=config.AT_DIVISOR)

    # Add sigmoid activation layer -
    x = __sigmoid_activation_layer(output=model.layers[-1].output, num_classes=config.NUM_CLASSES)
    model = Model(inputs=model.input, outputs=x)

    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, config.CP_SAVE_TAG + '.png'), show_shapes=True)

    return model


def segnet_2d(config):
    if (type(config) is not SegnetConfig):
        raise ValueError('config must be an instance of SegnetConfig')
    input_shape = config.IMG_SIZE
    #model = Segnet_v2(input_shape, config.NUM_CLASSES, depth=config.DEPTH, num_conv_layers=config.NUM_CONV_LAYERS, num_filters=config.NUM_FILTERS)
    model = Segnet(input_shape=input_shape, n_labels=config.NUM_CLASSES)
    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, config.CP_SAVE_TAG + '.png'), show_shapes=True)

    return model


def __sigmoid_activation_layer(output, num_classes):
    ''' Return sigmoid activation layer
        @param: output   The output of the previous layer
    '''

    # Initializing kernel weights to 1 and bias to 0.
    # i.e. without training, the output would be a sigmoid activation on each pixel of the input
    return Conv2D(num_classes, (1,1), activation='sigmoid', kernel_initializer=Ones(), bias_initializer=Zeros(), name='output_activation')(output)


if __name__ == '__main__':
    config = UNetMultiContrastConfig(create_dirs=False)
    config.INIT_UNET_2D_WEIGHTS = './test_data/unet_2d_fc_weights.004--0.8968.h5'

    unet_2d_multi_contrast(config)
