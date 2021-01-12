import logging
from copy import deepcopy

from keras import callbacks as kc

from medsegpy.utils import env

try:
    import wandb
    import wandb.wandb_run

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

    def __init__(self, period: int = 20, experiment="auto", **kwargs):
        """
        Args:
            period (int, optional): Logging period.
            experiment (`wandb.wandb_run.Run` | `str` | `None`): The experiment run.
                If ``"auto"``, a run will only be created if ``wandb.run`` is None.
                If ``None``, a run will be created.
            **kwargs: Options to pass to ``wandb.init()`` to create run. Ignored
                if ``experiment`` specified.
        """
        if not env.supports_wandb():
            raise ValueError(
                "Weights & Biases is not supported. "
                "Install package via `pip install wandb`. "
                "See documentation https://docs.wandb.com/ "
            )
        assert isinstance(experiment, wandb.wandb_run.Run) or experiment in ("auto", None)
        if (not wandb.run and experiment == "auto") or experiment is None:
            wandb.init(**kwargs)

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
