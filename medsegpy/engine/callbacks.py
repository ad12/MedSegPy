import logging
from copy import deepcopy

from keras import callbacks as kc

from medsegpy.utils import env

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no-cover
    wandb = None
    _WANDB_AVAILABLE = False


__all__ = ["lr_callback", "LossHistory", "WandBLogger"]

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
    """A Keras callback to log training history"""

    def on_train_begin(self, logs=None):
        self.val_losses = []
        self.losses = []
        self.epoch = []

    def on_epoch_end(self, epoch, logs=None):
        self.val_losses.append(logs.get("val_loss", float("nan")))
        self.losses.append(logs.get("loss"))
        self.epoch.append(epoch + 1)

        metrics = " - ".join(
            [
                "{}: {:0.4f}".format(k, v) if v >= 1e-3 else "{}: {:0.4e}".format(k, v)
                for k, v in logs.items()
            ]
        )
        logger.info("Epoch {} - {}".format(epoch + 1, metrics))


class WandBLogger(kc.Callback):
    """A Keras callback to log to weights and biases.

    Currently only supports logging scalars.
    """

    def __init__(self, period: int = 20):
        if not env.supports_wandb():
            raise ValueError(
                "Weights & Biases is not supported. "
                "Install package via `pip install wandb`. "
                "See documentation https://docs.wandb.com/ "
            )
        if not wandb.run:
            raise ValueError("Run `wandb.init(...) to configure the W&B run.")
        assert isinstance(period, int) and period > 0, "`period` must be int >0"

        self._period = period

    def on_train_begin(self, logs=None):
        self._step = 0

    def on_batch_end(self, batch_idx, logs=None):
        self._step += 1
        if not logs or self._step % self._period != 0:
            return

        wandb.log(logs, step=self._step)

    def on_epoch_end(self, epoch, logs=None):
        logs = deepcopy(logs)
        logs["epoch"] = epoch
        wandb.log(logs, step=self._step)
