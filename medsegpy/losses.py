import logging

import numpy as np
import scipy.special as sps
import tensorflow as tf
from keras import backend as K
from keras.callbacks import Callback
from keras.losses import binary_crossentropy

from medsegpy.loss.classification import DiceLoss
from medsegpy.utils import env

logger = logging.getLogger(__name__)

DICE_LOSS = ("dice", "sigmoid")
MULTI_CLASS_DICE_LOSS = ("multi_class_dice", "sigmoid")
AVG_DICE_LOSS = ("avg_dice", "sigmoid")
AVG_DICE_LOSS_SOFTMAX = ("avg_dice", "softmax")
AVG_DICE_NO_REDUCE = ("avg_dice_no_reduce", "sigmoid")
WEIGHTED_CROSS_ENTROPY_LOSS = ("weighted_cross_entropy", "softmax")
WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS = ("weighted_cross_entropy_sigmoid", "sigmoid")

BINARY_CROSS_ENTROPY_LOSS = ("binary_crossentropy", "softmax")

BINARY_CROSS_ENTROPY_SIG_LOSS = ("binary_crossentropy", "sigmoid")

FOCAL_LOSS = ("focal_loss", "sigmoid")
FOCAL_LOSS_GAMMA = 3.0

DICE_FOCAL_LOSS = ("dice_focal_loss", "sigmoid")
DICE_MEDIAN_LOSS = ("dice_median_loss", "sigmoid")

L2_LOSS = ("l2_loss", None)

CMD_LINE_SUPPORTED_LOSSES = [
    "DICE_LOSS",
    "MULTI_CLASS_DICE_LOSS",
    "AVG_DICE_LOSS",
    "AVG_DICE_LOSS_SOFTMAX",
    "AVG_DICE_NO_REDUCE",
    "WEIGHTED_CROSS_ENTROPY_LOSS",
    "WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS",
    "BINARY_CROSS_ENTROPY_LOSS",
    "BINARY_CROSS_ENTROPY_SIG_LOSS",
    "FOCAL_LOSS",
    "DICE_FOCAL_LOSS",
    "DICE_MEDIAN_LOSS",
    "L2_LOSS",
]


def build_loss(cfg):
    loss = cfg.LOSS
    num_classes = len(cfg.CATEGORIES)
    robust_loss_cls = cfg.ROBUST_LOSS_NAME
    robust_step_size = cfg.ROBUST_LOSS_STEP_SIZE

    if robust_loss_cls:
        reduction = "class"
    else:
        reduction = ""

    if isinstance(loss, str):
        try:
            loss = get_training_loss_from_str(loss)
        except (ValueError, AttributeError):
            pass
    loss = get_training_loss(
        loss,
        weights=cfg.CLASS_WEIGHTS,
        # Remove computation on the background class.
        remove_background=cfg.INCLUDE_BACKGROUND,
        reduce=reduction,
    )

    if not robust_loss_cls:
        return loss
    elif robust_loss_cls == "NaiveAdaRobLossComputer":
        return NaiveAdaRobLossComputer(
            criterion=loss, n_groups=num_classes, robust_step_size=robust_step_size
        )
    else:
        raise ValueError(f"{robust_loss_cls} not supported")


# TODO (arjundd): Add ability to exclude specific indices from loss function.
def get_training_loss_from_str(loss_str: str):
    loss_str = loss_str.upper()
    if loss_str == "DICE_LOSS":
        return DICE_LOSS
    elif loss_str == "MULTI_CLASS_DICE_LOSS":
        return MULTI_CLASS_DICE_LOSS
    elif loss_str == "AVG_DICE_LOSS" or loss_str == "AVG_DICE_LOSS_SOFTMAX":
        return AVG_DICE_LOSS
    elif loss_str == "AVG_DICE_NO_REDUCE":
        return AVG_DICE_NO_REDUCE
    elif loss_str == "WEIGHTED_CROSS_ENTROPY_LOSS":
        return WEIGHTED_CROSS_ENTROPY_LOSS
    elif loss_str == "WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS":
        return WEIGHTED_CROSS_ENTROPY_SIGMOID_LOSS
    elif loss_str == "BINARY_CROSS_ENTROPY_LOSS":
        return BINARY_CROSS_ENTROPY_LOSS
    elif loss_str == "BINARY_CROSS_ENTROPY_SIG_LOSS":
        return BINARY_CROSS_ENTROPY_SIG_LOSS
    elif loss_str == "FOCAL_LOSS":
        return FOCAL_LOSS
    elif loss_str == "DICE_FOCAL_LOSS":
        return DICE_FOCAL_LOSS
    elif loss_str == "DICE_MEDIAN_LOSS":
        return DICE_MEDIAN_LOSS
    elif loss_str == "L2_LOSS":
        return L2_LOSS
    else:
        raise ValueError("%s not supported" % loss_str)


def get_training_loss(loss, **kwargs):
    if loss == DICE_LOSS:
        return dice_loss
    elif loss == MULTI_CLASS_DICE_LOSS:
        return multi_class_dice_loss(**kwargs)
    elif loss == AVG_DICE_LOSS or loss == AVG_DICE_LOSS_SOFTMAX:
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
    elif loss == AVG_DICE_NO_REDUCE:
        kwargs.pop("reduce", None)
        kwargs["reduction"] = "none"
        return DiceLoss(**kwargs)
    elif loss == L2_LOSS:
        return l2_loss
    else:
        raise ValueError("Loss type not supported")


def _get_shape(x):
    """Returns shape of Keras tensor."""
    if hasattr(K, "int_shape"):
        return K.int_shape(x)
    else:
        return K.get_variable_shape(x)


def dice_focal_loss(y_true, y_pred):
    dsc = dice_loss(y_true, y_pred)
    fc = focal_loss(FOCAL_LOSS_GAMMA)(y_true, y_pred)

    return dsc + fc


# Dice function loss optimizer
def dice_loss(y_true, y_pred):
    """Computes class-agnostic dice."""
    szp = _get_shape(y_pred)

    img_len = np.product(szp[1:])

    if env.is_tf2():
        y_true = tf.dtypes.cast(y_true, y_pred.dtype)
    y_true = K.reshape(y_true, (-1, img_len))
    y_pred = K.reshape(y_pred, (-1, img_len))

    ovlp = K.sum(y_true * y_pred, axis=-1)

    mu = K.epsilon()
    dice = (2.0 * ovlp + mu) / (K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + mu)
    loss = 1 - dice

    return loss


def dice_median_loss(y_true, y_pred):
    """Get the median dice loss"""
    raise DeprecationWarning()
    lambda1 = 2
    mu = K.epsilon()

    szp = K.get_variable_shape(y_pred)
    img_len = szp[1] * szp[2] * szp[3]

    y_true = K.reshape(y_true, (-1, img_len))
    y_pred = K.reshape(y_pred, (-1, img_len))

    dsc = (2.0 * K.sum(y_true * y_pred, axis=-1) + mu) / (
        K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) + mu
    )
    dsc_mean = K.mean(dsc)
    dsc_std = K.std(dsc)
    bool_mask = tf.logical_and(
        K.greater_equal(dsc - dsc_mean + lambda1 * dsc_std, 0),
        K.less_equal(dsc - dsc_mean - lambda1 * dsc_std, 0),
    )

    binarize_dsc = tf.boolean_mask(dsc, bool_mask)

    loss = 1 - K.mean(binarize_dsc)

    return loss


def multi_class_dice_loss(
    weights=None, remove_background: bool = False, reduce="mean", use_numpy=False, **kwargs
):
    """Dice loss for multiple classes in softmax layer.

    Each class is treated individually.
    """
    use_weights = False
    if weights is not None:
        weights = K.variable(weights)
        use_weights = True

    def d_loss(y_true, y_pred):
        szp = _get_shape(y_pred)

        if env.is_tf2():
            y_true = tf.dtypes.cast(y_true, y_pred.dtype)
        y_true = K.reshape(y_true, (-1, szp[-1]))
        y_pred = K.reshape(y_pred, (-1, szp[-1]))

        ovlp = K.sum(y_true * y_pred, axis=0)

        mu = K.epsilon()
        dice = (2.0 * ovlp + mu) / (K.sum(y_true, axis=0) + K.sum(y_pred, axis=0) + mu)
        loss = 1 - dice

        if reduce == "class":
            return loss

        if use_weights:
            loss = weights * loss
            loss = K.sum(loss) / K.sum(weights)
        else:
            loss = K.mean(loss)

        return loss

    def d_loss_np(y_true, y_pred):
        szp = y_pred.shape

        y_true = np.reshape(y_true, (-1, szp[-1]))
        y_pred = np.reshape(y_pred, (-1, szp[-1]))

        ovlp = np.sum(y_true * y_pred, axis=0)

        mu = K.epsilon()
        dice = (2.0 * ovlp + mu) / (np.sum(y_true, axis=0) + np.sum(y_pred, axis=0) + mu)
        loss = 1 - dice

        if reduce == "class":
            return loss

        if use_weights:
            loss = weights * loss
            loss = np.sum(loss) / np.sum(weights)
        else:
            loss = np.mean(loss)

        return loss

    if use_numpy:
        return d_loss_np
    else:
        return d_loss


def avg_dice_loss(weights=None, remove_background: bool = False, **kwargs):
    use_weights = False
    if weights is not None:
        weights = np.asarray(weights)[np.newaxis, ...]
        weights = K.variable(weights)
        use_weights = True

    def d_loss(y_true, y_pred):
        szp = _get_shape(y_pred)
        c_dim = szp[-1]  # class dimension
        img_size = np.prod(szp[1:-1])  # vectorized image size

        # Keep batch and class dimensions.
        y_true = K.reshape(y_true, (-1, img_size, c_dim))
        y_pred = K.reshape(y_pred, (-1, img_size, c_dim))
        if remove_background:
            y_true = y_true[..., 1:]
            y_pred = y_pred[..., 1:]

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

    Usage:
    ```
    weights = np.array([0.5, 2]) # Class 1 at 0.5, class 2 2x the normal weights
    exclude_ind = 0  # exclude background from computation
    loss = weighted_categorical_crossentropy(weights=weights,
    exclude=exclude_ind)
    model.compile(loss=loss, optimizer='adam')
    ```
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
    ```
    weights = np.array([0.5, 2]) # Class 1 at 0.5, class 2 2x the normal weights
    loss = weighted_categorical_crossentropy(weights)
    model.compile(loss=loss,optimizer='adam')
    ```
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        # y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        szp = _get_shape(y_pred)
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
                M[i, j] * tf.multiply(prediction[:, i], ground_truth[:, j])
            )
    wass_dis_map = tf.add_n(pairwise_correlations)
    return wass_dis_map


M_tree_4 = np.array(
    [[0.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.6, 0.5], [1.0, 0.6, 0.0, 0.7], [1.0, 0.5, 0.7, 0.0]],
    dtype=np.float64,
)


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
        tf.multiply(tf.constant(M[0, :n_classes], dtype=tf.float64), one_hot), axis=1
    )
    true_pos = tf.reduce_sum(tf.multiply(true_pos, 1.0 - delta), axis=0)
    WGDL = 1.0 - (2.0 * true_pos) / (2.0 * true_pos + all_error)

    return tf.cast(WGDL, dtype=tf.float32)


def l2_loss(y_true, y_pred):
    """
    A basic L2 loss, which just compares the pixel values of two images or
    feature maps. The average L2 loss is returned, where the average is
    taken along the batch dimension.

    Args:
        y_true: The ground truth image or feature map.
        y_pred: The predicted image or feature map from the network.

    Returns:
       avg_l2_loss: A tensor, whose value is the average L2 loss.
    """
    square_diff = K.square(y_true - y_pred)
    ch_l2_loss = K.sum(square_diff, axis=(1, 2))
    avg_ch_l2 = K.mean(ch_l2_loss, axis=0)
    avg_l2_loss = K.mean(avg_ch_l2)
    return avg_l2_loss


class NaiveAdaRobLossComputer(Callback):
    """Handles adaptive robust class loss computer.

    Use `on_batch_begin` and `on_batch_end` to do things before/after the training batch.
    Note `on_batch_end` runs **before** validation when using `model.fit_generator`.
    Requires tf>=2.0 and eager mode execution.

    TODO:
        - Add adjustment
    """

    def __init__(self, criterion, n_groups, robust_step_size, stable=True):
        # TF2 requires __name__ attribute which is not set by default.
        self.__name__ = "NaiveAdaRobLossComputer"

        if not env.is_tf2():
            raise EnvironmentError(f"{self.__name__} only supported on tensorflow>=2.0")

        super().__init__()
        self.criterion = criterion

        self.n_groups = n_groups
        self.group_range = np.arange(self.n_groups, dtype=np.long)[..., np.newaxis]

        self.robust_step_size = robust_step_size
        logger.info(f"Using robust loss with inner step size {self.robust_step_size}")
        self.stable = stable

        # The following quantities are maintained/updated throughout training
        if self.stable:
            logger.info("Using numerically stabilized DRO algorithm")
            self.adv_probs_logits = np.zeros(self.n_groups)
        else:  # for debugging purposes
            logger.warning("Using original DRO algorithm")
            self.adv_probs = np.ones(self.n_groups) / self.n_groups

        self.training = None
        self.prev_training_state = None
        self._tmp = None  # Initialized in on_batch_begin (only used at train time)

    def loss(self, y_true, y_pred):
        # Get average classes losses (1D array - length C).
        # if not tf.executing_eagerly():
        #     raise RuntimeError(f"{self.__name__} does not support non-eager execution")

        group_losses = self.criterion(y_true, y_pred)  # Nx...xC
        if group_losses.ndim > 1:
            group_losses = tf.math.reduce_mean(group_losses, axis=range(0, group_losses.ndim - 1))
        # group_losses, group_counts = self.compute_group_avg(group_losses, y_true)

        # Compute overall loss.
        # The naive implementation does not do any sort of grouping.
        robust_loss = self.compute_robust_loss(group_losses)

        # Store for logging purposes.
        # Will be deleted after each batch.
        if self._tmp is not None:
            self._tmp["group_losses"] = group_losses.numpy()

        return robust_loss

    def compute_robust_loss(self, group_loss):
        # Requires eager execution
        if not tf.executing_eagerly():
            raise ValueError(f"{self.__name__} requires eager execution")

        assert (
            self.training is not None
        ), "`self.training` not initialized. Make sure this class is added as a callback"
        if self.training:
            # Update weighting if in training mode
            # This only works in eager mode.
            # TODO: Find solution for non-eager mode to speed up large-scale training
            adjusted_loss = group_loss if isinstance(group_loss, np.ndarray) else group_loss.numpy()
            logit_step = self.robust_step_size * adjusted_loss
            if self.stable:
                self.adv_probs_logits = self.adv_probs_logits + logit_step
            else:
                raise AssertionError("This branch has not been tested")
                # self.adv_probs = self.adv_probs * torch.exp(logit_step)
                # self.adv_probs = self.adv_probs / self.adv_probs.sum()

        if self.stable:
            adv_probs = (
                sps.softmax(self.adv_probs_logits)
                if isinstance(self.adv_probs_logits, np.ndarray)
                else K.softmax(self.adv_probs_logits, axis=-1)
            )
        else:
            adv_probs = self.adv_probs

        if isinstance(group_loss, np.ndarray):
            robust_loss = group_loss @ adv_probs
        else:
            robust_loss = K.sum(group_loss * adv_probs)
        return robust_loss

    def compute_group_avg(self, per_sample_losses, group_idx):
        """Compute average loss per group and counts of each group.

        Count is defined as the number of pixels corresponding to a group.
        Even for pixel-aggregate losses, such as dice, this is the default
        used.

        Args:
            per_sample_losses: Losses for each sample. Sample can either be batch
                or per-pixel.
        """
        group_count = tf.math.reduce_sum(group_idx, axis=range(1, group_idx.ndim - 1))
        if per_sample_losses.ndim == 2:  # NxC
            group_loss = tf.math.reduce_mean(per_sample_losses, axis=0)
        else:
            raise ValueError("`per_sample_losses` must have shape [N,C]")

        return group_loss, group_count

    def on_batch_begin(self, batch, logs=None):
        # Hacky way to turn on training before training batch in case it is off.
        self.training = True
        self._tmp = {}

    def on_batch_end(self, batch, logs=None):
        # Hacky way to turn off training when validation starts.
        # `fit_generator` executes `on_batch_end` before starting validation
        # This let's us turn off training mode during the validation period.
        self.training = False
        weighting = sps.softmax(self.adv_probs_logits)
        logs.update(
            {f"{self.__name__}/weights/class:{i}": weighting[i] for i in range(self.n_groups)}
        )
        group_losses = self._tmp["group_losses"]
        logs.update(
            {f"{self.__name__}/loss/class:{i}": group_losses[i] for i in range(self.n_groups)}
        )
        self._tmp = {}

    def on_test_begin(self, logs=None):
        self.prev_training_state = self.training
        self.training = False

    def on_test_end(self, logs=None):
        assert (
            self.prev_training_state is not None
        ), "`self.prev_training_state` should be set in `on_test_begin`"
        self.training = self.prev_training_state

    def __call__(self, y_true, y_pred):
        return self.loss(y_true, y_pred)
