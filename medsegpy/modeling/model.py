import warnings
from typing import Sequence

import tensorflow as tf
from keras.layers import ZeroPadding2D, ZeroPadding3D, Conv2D, Conv3D
from keras import backend as K
from keras.models import Model as _Model
from keras.utils.data_utils import GeneratorEnqueuer
from keras.utils.data_utils import OrderedEnqueuer
from keras.utils.generic_utils import Progbar
import numpy as np


class Model(_Model):
    """MedSegPy implementation of the Keras :class:`Model`.

    In addition to the traditional :class:`Model` functionality in Keras, this
    class offers the option of returning batches of inputs, outputs, and
    predictions using :meth:`inference_generator`.

    All models implemented in medsegpy should use this class as the base class.
    """

    def inference_generator(
        self,
        generator,
        steps = None,
        max_queue_size = 10,
        workers = 1,
        use_multiprocessing = False,
        verbose = 0
    ):
        """Generates predictions for the input samples from a data generator
        and returns inputs, ground truth, and predictions.

        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        Arguments:
            generator: Generator yielding batches of input samples
                or an instance of Sequence (keras.utils.Sequence)
                object in order to avoid duplicate data
                when using multiprocessing.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            max_queue_size: Maximum size for the generator queue.
            workers: Integer. Maximum number of processes to spin up
                when using process based threading.
                If unspecified, `workers` will default to 1. If 0, will
                execute the generator on the main thread.
            use_multiprocessing: If `True`, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            verbose: verbosity mode, 0 or 1.

        Returns:
            Numpy array(s) of inputs, outputs, predictions.

        # Raises
            ValueError: In case the generator yields
                data in an invalid format.
        """
        self._make_predict_function()

        steps_done = 0
        wait_time = 0.01
        all_outs = []
        all_xs = []
        all_ys = []
        is_sequence = isinstance(generator, Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning('Using a generator with `use_multiprocessing=True`'
                            ' and multiple workers may duplicate your data.'
                            ' Please consider using the`keras.utils.Sequence'
                            ' class.'))
        if steps is None:
            if is_sequence:
                steps = len(generator)
            else:
                raise ValueError('`steps=None` is only valid for a generator'
                                 ' based on the `keras.utils.Sequence` class.'
                                 ' Please specify `steps` or use the'
                                 ' `keras.utils.Sequence` class.')
        enqueuer = None

        try:
            if workers > 0:
                if is_sequence:
                    enqueuer = OrderedEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing,
                    )
                else:
                    enqueuer = GeneratorEnqueuer(
                        generator,
                        use_multiprocessing=use_multiprocessing,
                        wait_time=wait_time,
                    )
                enqueuer.start(workers=workers, max_queue_size=max_queue_size)
                output_generator = enqueuer.get()
            else:
                if is_sequence:
                    output_generator = iter(generator)
                else:
                    output_generator = generator

            if verbose == 1:
                progbar = Progbar(target=steps)

            while steps_done < steps:
                generator_output = next(output_generator)
                if isinstance(generator_output, tuple):
                    # Compatibility with the generators
                    # used for training.
                    if len(generator_output) == 2:
                        x, y = generator_output
                    elif len(generator_output) == 3:
                        x, y, _ = generator_output
                    else:
                        raise ValueError('Output of generator should be '
                                         'a tuple `(x, y, sample_weight)` '
                                         'or `(x, y)`. Found: ' +
                                         str(generator_output))
                else:
                    # Assumes a generator that only
                    # yields inputs and targets (not sample weights).
                    x, y = generator_output

                outs = self.predict_on_batch(x)
                xs = [x]
                ys = [y]
                if not isinstance(outs, list):
                    outs = [outs]

                if not all_outs:
                    for _ in outs:
                        all_outs.append([])
                if not all_xs:
                    for _ in xs:
                        all_xs.append([])
                if not all_ys:
                    for _ in ys:
                        all_ys.append([])

                for i, out in enumerate(outs):
                    all_outs[i].append(out)
                for i, x in enumerate(xs):
                    all_xs[i].append(x)
                for i, y in enumerate(ys):
                    all_ys[i].append(y)
                steps_done += 1
                if verbose == 1:
                    progbar.update(steps_done)

        finally:
            if enqueuer is not None:
                enqueuer.stop()

        if len(all_outs) == 1:
            if steps_done == 1:
                return all_xs[0][0], all_ys[0][0], all_outs[0][0]
            else:
                return np.concatenate(all_xs[0]), \
                       np.concatenate(all_ys[0]), \
                       np.concatenate(all_outs[0])
        if steps_done == 1:
            return [xs[0] for xs in all_xs], \
                   [ys[0] for ys in all_ys], \
                   [out[0] for out in all_outs]
        else:
            return [np.concatenate(xs) for xs in all_xs], \
                   [np.concatenate(ys) for ys in all_ys], \
                   [np.concatenate(out) for out in all_outs]


def get_primary_shape(x: tf.Tensor):
    """Get sizes of the primary dimensions of x.

    Primary dimensions of a tensor are all dimensions that do not correspond to
    the batch dimension `B` or the channel dimension `C`.

    Args:
        x (tf.Tensor): Shape Bx(...)xC (channels_last) or BxCx(...) (channels_first).  # noqa

    Returns:
        list: primary dimensions.
    """
    x_shape = x.shape.as_list()
    x_shape = x_shape[1:-1] \
        if K.image_data_format() == "channels_last" else x_shape[2:]

    return x_shape


def zero_pad_like(x: tf.Tensor, y: tf.Tensor, x_shape=None, y_shape=None):
    """Zero pads input (x) to size of target (y).

    Padding is symmetric when difference in dimension size is multiple of 2.
    Otherwise, the bottom padding is 1 larger than the top padding.
    Assumes channels are last dimension.

    Primary dimensions of a tensor are all dimensions that do not correspond to
    the batch dimension or the channel dimension.

    Args:
        x (tf.Tensor): Input tensor.
        y (tf.Tensor): Target tensor.
        x_shape (Sequence[int]): Expected shape of `x`. Required when primary
            `x` dimensions have sizes `None`.
        y_shape (Sequence[int]): Like `x_shape`, but for `y`.

    Returns:
        tf.Tensor: Zero-padded tensor
    """
    if not x_shape:
        x_shape = get_primary_shape(x)
    if not y_shape:
        y_shape = get_primary_shape(y)

    assert not any(s is None for s in x_shape)
    assert not any(s is None for s in y_shape)

    if x_shape == y_shape:
        return x
    diff = [y_s - x_s for x_s, y_s in zip(x_shape, y_shape)]
    assert all(d >= 0 for d in diff), (
        "x must be smaller than y in all dimensions"
    )

    if len(diff) == 2:
        padder = ZeroPadding2D
    elif len(diff) == 3:
        padder = ZeroPadding3D
    else:
        raise ValueError("Zero padding available for 2D or 3D images only")

    padding = [d//2 if d % 2 == 0 else (d//2, d//2 + 1) for d in diff]
    x = padder(padding)(x)
    return x


def add_sem_seg_activation(
    x: tf.Tensor,
    num_classes: int,
    activation: str="sigmoid",
    conv_type=None,
    kernel_initializer=None,
    seed = None,
) -> tf.Tensor:
    """Standardized output layer for semantic segmentation using 1x1 conv.

    Args:
        x (tf.Tensor): Input tensor.
        num_classes (int): Number of classes
        activation (str, optional): Activation type. Typically `'sigmoid'` or
            `'softmax'`.
        conv_type: Either `Conv2D` or `Conv3D`.
        kernel_initializer: Kernel initializer accepted by
            `Conv2D` or `Conv3D`.
        seed (int, optional): Kernel intialization seed. Ignored if
            `kernel_initializer` is a config dict.
    """

    # Initializing kernel weights to 1 and bias to 0.
    # i.e. without training, the x would be a sigmoid activation on each
    # pixel of the input
    if not conv_type:
        conv_type = Conv2D
    else:
        assert conv_type in [Conv2D, Conv3D]
    if not kernel_initializer:
        kernel_initializer = {
            "class_name": "glorot_uniform",
            "config": {"seed": seed},
        }
    elif isinstance(kernel_initializer, str):
        kernel_initializer = {
            "class_name": kernel_initializer,
            "config": {"seed": seed},
        }

    x = conv_type(
        num_classes,
        1,
        activation=activation,
        kernel_initializer=kernel_initializer,
        name="output_activation",
    )(x)
    return x
