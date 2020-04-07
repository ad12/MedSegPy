import logging

from keras.optimizers import Adam
from keras import callbacks as kc

from medsegpy.config import Config
from .lr_scheduler import step_decay
from .optimizer import AdamAccumulate

__all__ = ["build_lr_scheduler", "build_optimizer"]

logger = logging.getLogger("msk_seg_networks.{}".format(__name__))


def build_optimizer(config: Config):
    """Build optimizer from config.

    Currently supports :class:`Adam` or :class:`AdamAccumulate` optimizers.

    Args:
        config (Config): A config to read parameters from.

    Returns:
        A Keras-compatible optimizer.
    """
    if config.NUM_GRAD_STEPS == 1:
        optimizer = Adam(
            lr=config.INITIAL_LEARNING_RATE,
            beta_1=0.99,
            beta_2=0.995,
            epsilon=1e-8,
            decay=config.ADAM_DECAY,
            amsgrad=config.USE_AMSGRAD
        )
    elif config.NUM_GRAD_STEPS > 1:
        logger.info(
            "Accumulating gradient over {} steps".format(config.NUM_GRAD_STEPS)
        )
        optimizer = AdamAccumulate(
            lr=config.INITIAL_LEARNING_RATE,
            beta_1=0.99,
            beta_2=0.995, epsilon=1e-8,
            decay=config.ADAM_DECAY,
            amsgrad=config.USE_AMSGRAD,
            accum_iters=config.NUM_GRAD_STEPS
        )
    else:
        raise ValueError("config.NUM_GRAD_STEPS must be >= 1")

    return optimizer


def build_lr_scheduler(config: Config) -> kc.Callback:
    """Build learning rate scheduler.

    Supports "StepDecay" and "ReduceLROnPlateau"

        Args:
        config (Config): A config to read parameters from.

    Returns:
        :class:`keras.callback.LearningRateScheduler`

    Usage:
        >>> callbacks = []  # list of callbacks to be used sith `fit_generator`
        >>> scheduler = build_lr_scheduler(...)
        >>> callbacks.append(scheduler)
    """
    name = config.LR_SCHEDULER_NAME
    if name == "StepDecay":
        scheduler_func = step_decay(
            initial_lr=config.INITIAL_LEARNING_RATE,
            min_lr=config.MIN_LEARNING_RATE,
            drop_factor=config.DROP_FACTOR,
            drop_rate=config.DROP_RATE,
        )
        return kc.LearningRateScheduler(scheduler_func)
    elif name == "ReduceLROnPlateau":
        if config.LR_PATIENCE <= 0:
            raise ValueError(
                "LR patience must be >= 0. Got {}".format(config.LR_PATIENCE)
            )
        return kc.ReduceLROnPlateau(
            factor=config.DROP_FACTOR,
            patience=config.LR_PATIENCE,
            min_delta=config.LR_MIN_DELTA,
            cooldown=config.LR_COOLDOWN
        )
    else:
        raise ValueError(
            "LR scheduler {} not supported".format(config.LR_SCHEDULER_NAME)
        )
