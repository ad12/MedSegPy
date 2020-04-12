import logging
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))

DICE_LOSS = ('dice', 'sigmoid')
MULTI_CLASS_DICE_LOSS = ("multi_class_dice", "sigmoid")
AVG_DICE_LOSS = ("avg_dice", "sigmoid")

WEIGHTED_CROSS_ENTROPY_LOSS = ('weighted_cross_entropy', 'softmax')
WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS = ('weighted_cross_entropy_sigmoid', 'sigmoid')

BINARY_CROSS_ENTROPY_LOSS = ('binary_crossentropy', 'softmax')

BINARY_CROSS_ENTROPY_SIG_LOSS = ('binary_crossentropy', 'sigmoid')

FOCAL_LOSS = ('focal_loss', 'sigmoid')
FOCAL_LOSS_GAMMA = 3.0

DICE_FOCAL_LOSS = ('dice_focal_loss', 'sigmoid')
DICE_MEDIAN_LOSS = ('dice_median_loss', 'sigmoid')

CMD_LINE_SUPPORTED_LOSSES = ['DICE_LOSS',
                             "MULTI_CLASS_DICE_LOSS",
                             "AVG_DICE_LOSS",
                             'WEIGHTED_CROSS_ENTROPY_LOSS',
                             'WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS',
                             'BINARY_CROSS_ENTROPY_LOSS',
                             'BINARY_CROSS_ENTROPY_SIG_LOSS',
                             'FOCAL_LOSS',
                             'DICE_FOCAL_LOSS',
                             'DICE_MEDIAN_LOSS',
                             ]


# TODO (arjundd): Add ability to exclude specific indices from loss function computation
def get_training_loss_from_str(loss_str: str):
    loss_str = loss_str.upper()
    if loss_str == 'DICE_LOSS':
        return DICE_LOSS
    elif loss_str == "MULTI_CLASS_DICE_LOSS":
        return MULTI_CLASS_DICE_LOSS
    elif loss_str == "AVG_DICE_LOSS":
        return AVG_DICE_LOSS
    elif loss_str == 'WEIGHTED_CROSS_ENTROPY_LOSS':
        return WEIGHTED_CROSS_ENTROPY_LOSS
    elif loss_str == 'WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS':
        return WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS
    elif loss_str == 'BINARY_CROSS_ENTROPY_LOSS':
        return BINARY_CROSS_ENTROPY_LOSS
    elif loss_str == 'BINARY_CROSS_ENTROPY_SIG_LOSS':
        return BINARY_CROSS_ENTROPY_SIG_LOSS
    elif loss_str == 'FOCAL_LOSS':
        return FOCAL_LOSS
    elif loss_str == 'DICE_FOCAL_LOSS':
        return DICE_FOCAL_LOSS
    elif loss_str == 'DICE_MEDIAN_LOSS':
        return DICE_MEDIAN_LOSS
    else:
        raise ValueError('%s not supported' % loss_str)


def get_training_loss(loss, **kwargs):
    if loss == DICE_LOSS:
        return dice_loss
    elif loss == MULTI_CLASS_DICE_LOSS:
        return multi_class_dice_loss(**kwargs)
    elif loss == AVG_DICE_LOSS:
        return avg_dice_loss(**kwargs)
    elif loss == WEIGHTED_CROSS_ENTROPY_LOSS:
        return weighted_categorical_crossentropy(**kwargs)
    elif loss == BINARY_CROSS_ENTROPY_LOSS:
        return binary_crossentropy
    elif loss == BINARY_CROSS_ENTROPY_SIG_LOSS:
        return binary_crossentropy
    elif loss == FOCAL_LOSS:
        return focal_loss(**kwargs)
    elif loss == WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS:
        return weighted_categorical_crossentropy_sigmoid(**kwargs)
    elif loss == DICE_FOCAL_LOSS:
        return dice_focal_loss
    elif loss == DICE_MEDIAN_LOSS:
        return dice_median_loss
    else:
        raise ValueError("Loss type not supported")


def dice_focal_loss(y_true, y_pred):
    dsc = dice_loss(y_true, y_pred)
    fc = focal_loss(FOCAL_LOSS_GAMMA)(y_true, y_pred)

    return dsc + fc


# Dice function loss optimizer
def dice_loss(y_true, y_pred):
    szp = K.get_variable_shape(y_pred)
    img_len = szp[1] * szp[2] * szp[3]

    y_true = K.reshape(y_true, (-1, img_len))
    y_pred = K.reshape(y_pred, (-1, img_len))

    ovlp = K.sum(y_true * y_pred, axis=-1)

    mu = K.epsilon()
    dice = (2.0 * ovlp + mu) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + mu)
    loss = 1 - dice

    return loss


def dice_median_loss(y_true, y_pred):
    """Get the median dice loss"""
    lambda1 = 2
    mu = K.epsilon()
    
    szp = K.get_variable_shape(y_pred)
    img_len = szp[1] * szp[2] * szp[3]
    
    y_true = K.reshape(y_true, (-1, img_len))
    y_pred = K.reshape(y_pred, (-1, img_len))
    
    dsc = (2.0 * K.sum(y_true * y_pred, axis=-1) + mu) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + mu)
    dsc_mean = K.mean(dsc)
    dsc_std = K.std(dsc)
    bool_mask = tf.logical_and(K.greater_equal(dsc - dsc_mean + lambda1*dsc_std, 0),
                               K.less_equal(dsc - dsc_mean - lambda1*dsc_std, 0))
    
    binarize_dsc = tf.boolean_mask(dsc, bool_mask)

    loss = 1 - K.mean(binarize_dsc)

    return loss


def multi_class_dice_loss(
    weights=None,
    remove_background: bool=False,
    **kwargs
):
    """Dice loss for multiple classes in softmax layer.

    Each class is treated individually.
    """
    use_weights = False
    if weights:
        weights = K.variable(weights)
        use_weights = True

    def d_loss(y_true, y_pred):
        szp = K.get_variable_shape(y_pred)

        y_true = K.reshape(y_true, (-1, szp[-1]))
        y_pred = K.reshape(y_pred, (-1, szp[-1]))

        ovlp = K.sum(y_true * y_pred, axis=0)

        mu = K.epsilon()
        dice = (2.0 * ovlp + mu) / (K.sum(y_true, axis=0) + K.sum(y_pred, axis=0) + mu)
        loss = 1 - dice

        if use_weights:
            loss = weights * loss
            loss = K.sum(loss) / K.sum(weights)
        else:
            loss = K.mean(loss)

        return loss

    return d_loss


def avg_dice_loss(weights=None, **kwargs):
    use_weights = False
    if weights:
        weights = np.asarray(weights)[np.newaxis, ...]
        weights = K.variable(weights)
        use_weights = True

    def d_loss(y_true, y_pred):
        szp = K.get_variable_shape(y_pred)

        # Keep batch and class dimensions.
        y_true = K.reshape(y_true, (szp[0], -1, szp[-1]))
        y_pred = K.reshape(y_pred, (szp[0], -1, szp[-1]))

        ovlp = K.sum(y_true * y_pred, axis=1)

        mu = K.epsilon()
        dice = (2.0 * ovlp + mu) / (K.sum(y_true, axis=1) + K.sum(y_pred, axis=1) + mu)
        loss = 1 - dice

        if use_weights:
            loss = weights * loss
            loss = K.mean(K.sum(loss, axis=-1) / K.sum(weights))
        else:
            loss = K.mean(loss)

        return loss

    return d_loss


def weighted_categorical_crossentropy(weights, **kwargs):
    """A weighted version of Keras categorical_crossentropy

    :param: weights: Numpy array of shape (C,) where C is the number of classes

    Use Case:
        weights = np.array([0.5, 2]) # Class one at 0.5, class 2 2x the normal weights
        exclude_ind = 0  # exclude background from computation
        loss = weighted_categorical_crossentropy(weights=weights, exclude=exclude_ind)
        model.compile(loss=loss, optimizer='adam')
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


def weighted_categorical_crossentropy_sigmoid(weights):
    """
    A weighted version of Keras categorical_crossentropy

    @:param: weights: numpy array of shape (C,) where C is the number of classes

    Use Case:
        weights = np.array([0.5, 2]) # Class one at 0.5, class 2 2x the normal weights
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        szp = K.get_variable_shape(y_pred)
        img_len = szp[1] * szp[2] * szp[3]

        y_true = K.reshape(y_true, (-1, img_len))
        y_pred = K.reshape(y_pred, (-1, img_len))
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc
        loss = (1 - y_true) * K.log(1 - y_pred) * weights[0] + y_true * K.log(y_pred) * weights[1]
        loss = -K.mean(loss)
        return loss

    return loss


def focal_loss(gamma=FOCAL_LOSS_GAMMA):
    """
    Focal loss as implemented by facebook

    formula is -(1 - pt)^gamma * log(pt),   pt = p if y=1
                                            pt = 1-p if y=0

    gamma = 3

    y_true values must be 0s and 1s

    @:param: weights: numpy array of shape (C,) where C is the number of classes
    """

    def f_loss(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calculate pt
        # note that ~y_true = 1 - y_true
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)

        loss_val = -K.mean(K.pow((1 - pt), gamma) * K.log(pt)) * 100

        return loss_val

    return f_loss


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
    # logger.info("shape of M", M.shape, "unstacked labels", unstack_labels,
    #       "unstacked pred" ,unstack_pred)
    # W is a weighting sum of all pairwise correlations (pred_ci x labels_cj)
    pairwise_correlations = []
    for i in range(n_classes):
        for j in range(n_classes):
            pairwise_correlations.append(
                M[i, j] * tf.multiply(prediction[:, i], ground_truth[:, j]))
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


M_tree_4 = np.array([[0., 1., 1., 1., ],
                     [1., 0., 0.6, 0.5],
                     [1., 0.6, 0., 0.7],
                     [1., 0.5, 0.7, 0.]], dtype=np.float64)


def generalised_wasserstein_dice_loss(y_true, y_predicted):
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

    ground_truth = tf.cast(tf.reshape(y_true, (-1, n_classes)), dtype=tf.int64)
    pred_proba = tf.cast(tf.reshape(y_predicted, (-1, n_classes)), dtype=tf.float64)

    # M = tf.cast(M, dtype=tf.float64)
    # compute disagreement map (delta)
    M = M_tree_4
    # logger.info("M shape is ", M.shape, pred_proba, one_hot)
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
