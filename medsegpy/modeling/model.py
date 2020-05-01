import warnings

import numpy as np
from keras import utils as k_utils
from keras.models import Model as _Model
from keras.utils.data_utils import GeneratorEnqueuer, OrderedEnqueuer
from keras.utils.generic_utils import Progbar

__all__ = ["Model"]


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
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
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
        is_sequence = isinstance(generator, k_utils.Sequence)
        if not is_sequence and use_multiprocessing and workers > 1:
            warnings.warn(
                UserWarning(
                    "Using a generator with `use_multiprocessing=True`"
                    " and multiple workers may duplicate your data."
                    " Please consider using the`keras.utils.Sequence"
                    " class."
                )
            )
        if steps is None:
            if is_sequence:
                steps = len(generator)
            else:
                raise ValueError(
                    "`steps=None` is only valid for a generator"
                    " based on the `keras.utils.Sequence` class."
                    " Please specify `steps` or use the"
                    " `keras.utils.Sequence` class."
                )
        enqueuer = None

        try:
            if workers > 0:
                if is_sequence:
                    enqueuer = OrderedEnqueuer(
                        generator, use_multiprocessing=use_multiprocessing
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
                        x_raw = None
                    elif len(generator_output) == 3:
                        x, y, x_raw = generator_output
                        assert isinstance(x_raw, np.ndarray)
                    else:
                        raise ValueError(
                            "Output of generator should be "
                            "a tuple `(x, y, sample_weight)` "
                            "or `(x, y)`. Found: " + str(generator_output)
                        )
                else:
                    # Assumes a generator that only
                    # yields inputs and targets (not sample weights).
                    x, y = generator_output
                    x_raw = None

                outs = self.predict_on_batch(x)
                xs = [x_raw if x_raw is not None else x]
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
                return (
                    np.concatenate(all_xs[0]),
                    np.concatenate(all_ys[0]),
                    np.concatenate(all_outs[0]),
                )
        if steps_done == 1:
            return (
                [xs[0] for xs in all_xs],
                [ys[0] for ys in all_ys],
                [out[0] for out in all_outs],
            )
        else:
            return (
                [np.concatenate(xs) for xs in all_xs],
                [np.concatenate(ys) for ys in all_ys],
                [np.concatenate(out) for out in all_outs],
            )
