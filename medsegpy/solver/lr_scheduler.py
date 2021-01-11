"""Learning rate schedulers.

Usage:
    >>> callbacks = []  # list of callbacks to be used sith `fit_generator`
    >>> scheduler = step_decay(...)
    >>> callbacks.append(keras.callback.LearningRateScheduler(scheduler))
"""


def step_decay(initial_lr, min_lr, drop_factor, drop_rate):
    """Learning rate drops by factor of `drop_factor` every `drop_rate` epochs.

    For legacy purposes, the first drop occurs after `drop_rate - 1` epochs.
    For example, if `drop_rate = 3`, the first decay will occur after 2 epochs.
    Subsequently, the learning rate will drop every 3 epochs.

    Args:
        initial_lr: initial learning rate (default = 1e-4)
        min_lr: minimum learning rate (default = None)
        drop_factor: factor to drop (default = 0.8)
        drop_rate: rate of learning rate drop (default = 1.0 epochs)

    Returns:
        func: To be used with :class`keras.callbacks.LearningRateScheduler`
    """
    initial_lr = initial_lr
    drop_factor = drop_factor
    drop_rate = drop_rate
    min_lr = min_lr

    def callback(epoch):
        import math

        lrate = initial_lr * math.pow(drop_factor, math.floor((1 + epoch) / drop_rate))
        if lrate < min_lr:
            lrate = min_lr

        return lrate

    return callback
