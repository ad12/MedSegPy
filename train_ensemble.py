import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

from keras import Model
from keras.initializers import Constant
from keras.initializers import glorot_uniform
from keras.layers import Input, Conv2D, Concatenate
from keras.utils import plot_model

from config import EnsembleUDSConfig
from deeplab_2d.deeplab_model import Deeplabv3
from glob_constants import SEED
from segnet_2d.segnet import Segnet, Segnet_v2
from unet_2d.unet_model import unet_2d_model

from oai_train import train_model

DEEPLAB_WEIGHTS = ''
SEGNET_WEIGHTS = ''
UNET_WEIGHTS = ''


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
    deeplab_model = Deeplabv3(weights=None, input_shape=input_shape, classes=num_classes, backbone='xception',
                              OS=16, dil_rate_input=(2,4,6))
    deeplab_model.load_weights(DEEPLAB_WEIGHTS, by_name=True)
    deeplab_model.trainable = False
    x = deeplab_model.input

    # Unet
    unet_model = unet_2d_model(input_size=input_shape, input_tensor=x)
    unet_model.load_weights(UNET_WEIGHTS, by_name=True)
    unet_model.trainable = False

    # Segnet
    segnet_model = Segnet(input_tensor=x, n_labels=num_classes)
    print('Loaded Segnet')
    segnet_model.load_weights(SEGNET_WEIGHTS)
    segnet_model.trainable = False

    model = combine_models(x, [unet_model, deeplab_model, segnet_model], ensemble_name='ensemble_uds')
    plot_model(model, os.path.join(config.PLOT_MODEL_PATH, config.CP_SAVE_TAG + '.png'), show_shapes=True)

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
    x = Conv2D(num_classes, (1, 1), name='%s_conv' % ensemble_name, activation='sigmoid',
               kernel_initializer=Constant(value=1.0 / len(models)))(x)

    model = Model(inputs=x_input, outputs=x)
    for layer in model.layers[:-1]:
        layer.trainable = False

    model.summary()
    return model


if __name__ == '__main__':
    config = EnsembleUDSConfig()
    model = ensemble_uds(config)

    train_model(config, model=model)