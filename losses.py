import numpy as np
from enum import Enum
from keras import backend as K


class Loss(Enum):
    DICE = (0, 'sigmoid')
    WEIGHTED_CROSS_ENTROPY = (1, 'softmax')


def get_training_loss(loss, weights=None):
    if loss == Loss.DICE:
        return dice_loss
    elif loss == Loss.WEIGHTED_CROSS_ENTROPY:
        if weights is None:
            raise ValueError("Weights must be specified to initialize weighted_cross_entropy")
        return weighted_categorical_crossentropy(weights)
    else:
        raise ValueError("Loss type not supported")


# Dice function loss optimizer
def dice_loss(y_true, y_pred):

    szp = K.get_variable_shape(y_pred)
    img_len = szp[1]*szp[2]*szp[3]

    y_true = K.reshape(y_true,(-1,img_len))
    y_pred = K.reshape(y_pred,(-1,img_len))

    ovlp = K.sum(y_true*y_pred,axis=-1)

    mu = K.epsilon()
    dice = (2.0 * ovlp + mu) / (K.sum(y_true,axis=-1) + K.sum(y_pred,axis=-1) + mu)
    loss = 1 - dice

    return loss

# Dice function loss optimizer
# During test time since it includes a discontinuity
def dice_loss_test(y_true, y_pred):
    
    recon = np.squeeze(y_true)
    pred = np.squeeze(y_pred)
    y_pred = (y_pred > 0.05)*y_pred

    szp = y_pred.shape
    img_len = szp[1]*szp[2]*szp[3]

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    ovlp = np.sum(y_true*y_pred,axis=-1)

    mu = 1e-07
    dice = (2.0 * ovlp + mu) / (np.sum(y_true,axis=-1) + np.sum(y_pred,axis=-1) + mu)

    return dice


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of Keras categorical_crossentropy

    @:param: weights: numpy array of shape (C,) where C is the number of classes

    Use Case:
        weights = np.array([0.5,2]) # Class one at 0.5, class 2 2x the normal weights
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
