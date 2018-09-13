import numpy as np
from enum import Enum
from keras import backend as K
import tensorflow as tf

# Losses
DICE_LOSS = ('dice', 'sigmoid')
WEIGHTED_CROSS_ENTROPY_LOSS = ('weighted_cross_entropy', 'softmax')


def get_training_loss(loss, weights=None):
    if loss == DICE_LOSS:
        return dice_loss
    elif loss == WEIGHTED_CROSS_ENTROPY_LOSS:
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
        print(K.shape(y_true))
        print(K.shape(y_pred))
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def wasserstein_disagreement_map(prediction, ground_truth, M):
    """
    Function to calculate the pixel-wise Wasserstein distance between the
    flattened pred_proba and the flattened labels (ground_truth) with respect
    to the distance matrix on the label space M.
    :param prediction: the logits after softmax
    :param ground_truth: segmentation ground_truth
    :param M: distance matrix on the label space
    :return: the pixelwise distance map (wass_dis_map)
    """
    # pixel-wise Wassertein distance (W) between flat_pred_proba and flat_labels
    # wrt the distance matrix on the label space M
    n_classes = K.int_shape(prediction)[-1]
    # unstack_labels = tf.unstack(ground_truth, axis=-1)
    ground_truth = tf.cast(ground_truth, dtype=tf.float64)
    # unstack_pred = tf.unstack(prediction, axis=-1)
    prediction = tf.cast(prediction, dtype=tf.float64)
    # print("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(prediction[:,i], ground_truth[:,j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


M_tree_4 = np.array([[0., 1., 1., 1.,],
                     [1., 0., 0.6, 0.5],
                     [1., 0.6, 0., 0.7],
                     [1., 0.5, 0.7, 0.]], dtype=np.float64)


def generalised_wasserstein_dice_loss(y_true, y_predicted ):


    """
    Function to calculate the Generalised Wasserstein Dice Loss defined in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)
    :param prediction: the logits (before softmax)
    :param ground_truth: the segmentation ground_truth
    :param weight_map:
    :return: the loss
    """
    # apply softmax to pred scores
    n_classes = K.int_shape(y_predicted)[-1]


    ground_truth = tf.cast(tf.reshape(y_true,(-1,n_classes)), dtype=tf.int64)
    pred_proba = tf.cast(tf.reshape(y_predicted,(-1,n_classes)), dtype=tf.float64)

    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree_4
    # print("M shape is ", M.shape, pred_proba, one_hot)
    delta = wasserstein_disagreement_map(pred_proba, ground_truth, M)
    # compute generalisation of all error for multi-class seg
    all_error = tf.reduce_sum(delta)
    # compute generalisation of true positives for multi-class seg
    one_hot = tf.cast(ground_truth, dtype=tf.float64)
    true_pos = tf.reduce_sum(
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot),
        axis=1)
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1. - delta), axis=0)
    WGDL = 1. - (2. * true_pos) / (2. * true_pos + all_error)

    return tf.cast(WGDL, dtype=tf.float32)