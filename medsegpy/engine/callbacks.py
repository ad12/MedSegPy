import logging

from keras import callbacks as kc

__all__ = ["lr_callback",
           "LossHistory"]

logger = logging.getLogger(__name__)


def lr_callback(optimizer):
    """Wrapper for learning rate tensorflow metric.

    Args:
        optimizer: Optimizer used for training.

    Returns:
        func: To be wrapped in metric or callback.
    """

    def lr(y_true, y_pred):
        return optimizer.lr

    return lr


class LossHistory(kc.Callback):
    """A Keras callback to log training history
    """

    def on_train_begin(self, logs = {}):
        self.val_losses = []
        self.losses = []
        self.epoch = []

    def on_epoch_end(self, epoch, logs = {}):
        self.val_losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('loss'))
        self.epoch.append(epoch + 1)

        metrics = " - ".join(["{}: {:0.4f}".format(k, v)
                              if v >= 1e-3 else "{}: {:0.4e}".format(k, v)
                              for k, v in logs.items()])
        logger.info("Epoch {} - {}".format(epoch + 1, metrics))
