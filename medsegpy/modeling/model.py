import warnings

import numpy as np
import tensorflow as tf
from keras import utils as k_utils
from keras.models import Model as _Model
from keras.utils.data_utils import GeneratorEnqueuer, OrderedEnqueuer
from keras.utils.generic_utils import Progbar

from medsegpy.utils import env

if env.is_tf2():
    from tensorflow.python.keras.engine.training import concat
    from tensorflow.python.keras import callbacks as callbacks_module
    from tensorflow.python.keras.engine import data_adapter
    from tensorflow.python.util import nest
    from tensorflow.python.keras.utils import tf_utils
else:
    concat = None
    callbacks_module = None
    data_adapter = None
    nest = None
    tf_utils = None

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
        return self.inference_generator_static(
            self, generator, steps, max_queue_size, workers, use_multiprocessing, verbose
        )

    @classmethod
    def inference_generator_static(
        cls,
        model,
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
            model: The Keras model.
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
        # Issue #8: Remove this check when shuffle=True is supported.
        if (
            hasattr(generator, "shuffle")
            and isinstance(generator.shuffle, bool)
            and generator.shuffle
        ):
            raise ValueError(
                "Shuffling in generator is not supported. "
                "Set `generator.shuffle=False`."
            )

        if env.is_tf2():
            # TODO (TF2.X): Update when dataloaders migrate from keras.Sequence -> tf.data
            return cls._inference_generator_tf2(
                model=model,
                x=generator,
                steps=steps,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=verbose,
            )
        else:
            return model._inference_generator_tf1(
                generator=generator,
                steps=steps,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                verbose=verbose,
            )


    def _inference_generator_tf1(
        self,
        generator,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        verbose=0,
    ):
        """Inference generator for TensorFlow 1."""
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
                x, y, x_raw = _extract_inference_inputs(generator_output)

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

    @staticmethod
    def _inference_generator_tf2(
        model: _Model,
        x,
        batch_size=None,
        verbose=0,
        steps=None,
        callbacks=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False
    ):
        """Inference generator for TensorFlow 2."""
        outputs = []
        xs = []
        ys = []
        with model.distribute_strategy.scope():
            # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
            data_handler = data_adapter.DataHandler(
                x=x,
                batch_size=batch_size,
                steps_per_epoch=steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=model,
                steps_per_execution=model._steps_per_execution
            )

            # Container that configures and calls `tf.keras.Callback`s.
            if not isinstance(callbacks, callbacks_module.CallbackList):
                callbacks = callbacks_module.CallbackList(
                    callbacks,
                    add_history=True,
                    add_progbar=verbose != 0,
                    model=model,
                    verbose=verbose,
                    epochs=1,
                    steps=data_handler.inferred_steps
                )

            # predict_function = model.make_predict_function()
            model._predict_counter.assign(0)
            callbacks.on_predict_begin()
            for _, iterator in data_handler.enumerate_epochs():  # Single epoch.
                # iterator = peekable(iterator)
                with data_handler.catch_stop_iteration():
                    for step in data_handler.steps():
                        callbacks.on_predict_batch_begin(step)
                        batch_x, batch_y, batch_x_raw = _extract_inference_inputs(next(iterator))
                        # tmp_batch_outputs = predict_function(iterator)
                        tmp_batch_outputs = model.predict(batch_x)
                        if data_handler.should_sync:
                            context.async_wait()
                        batch_outputs = tmp_batch_outputs  # No error, now safe to assign.
                        
                        if batch_x_raw is not None:
                            batch_x = batch_x_raw
                        for batch, running in zip([batch_x, batch_y, batch_outputs], [xs, ys, outputs]):
                            nest.map_structure_up_to(
                                batch,
                                lambda x, batch_x: x.append(batch_x),
                                running, 
                                batch,
                            )

                        end_step = step + data_handler.step_increment
                        callbacks.on_predict_batch_end(end_step, {'outputs': batch_outputs})
                callbacks.on_predict_end()

            xs = [tf_utils.to_numpy_or_python_type(_x) for _x in xs]
            ys = [tf_utils.to_numpy_or_python_type(_y) for _y in ys]
            outputs = [tf_utils.to_numpy_or_python_type(_o) for _o in outputs]
            all_xs = nest.map_structure_up_to(batch_x, np.concatenate, xs)
            all_ys = nest.map_structure_up_to(batch_y, np.concatenate, ys)
            all_outputs = nest.map_structure_up_to(batch_outputs, np.concatenate, outputs)
            return all_xs, all_ys, all_outputs

            # all_xs = nest.map_structure_up_to(batch_x, concat, xs)
            # all_ys = nest.map_structure_up_to(batch_y, concat, ys)
            # all_outputs = nest.map_structure_up_to(batch_outputs, concat, outputs)
            # return (
            #     tf_utils.to_numpy_or_python_type(all_xs),
            #     tf_utils.to_numpy_or_python_type(all_ys),
            #     tf_utils.to_numpy_or_python_type(all_outputs),
            # )

def _extract_inference_inputs(inputs):
    def check_type(x):
        if env.is_tf2():
            return isinstance(x, tf.Tensor)
        else:
            return isinstance(x, np.ndarray)

    if isinstance(inputs, tuple):
        # Compatibility with the generators
        # used for training.
        if len(inputs) == 2:
            x, y = inputs
            x_raw = None
        elif len(inputs) == 3:
            x, y, x_raw = inputs
            assert check_type(x_raw)
        else:
            raise ValueError(
                "Output of generator should be "
                "a tuple `(x, y, x_raw)` "
                "or `(x, y)`. Found: " + str(inputs)
            )
    else:
        # Assumes a generator that only
        # yields inputs and targets (not sample weights).
        x, y = inputs
        x_raw = None

    return x, y, x_raw