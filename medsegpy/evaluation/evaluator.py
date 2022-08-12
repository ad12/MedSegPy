"""Dataset evaluator.

Adopted from Facebook's detectron2.
https://github.com/facebookresearch/detectron2
"""
import datetime
import logging
import time
from typing import Sequence, Union

from medsegpy.data import DataLoader
from medsegpy.data.im_gens import Generator, GeneratorState
from medsegpy.utils.logger import log_every_n_seconds


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the
        inputs/outputs.

    This class will accumulate information of the inputs/outputs
        (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs):
        """
        Process an input/output pair.

        Args:
            scan_id: the scan id corresponding to the input/output
            inputs (List[Dict]]: the inputs that are used to call the model.
                Can also contain scan specific fields. These fields
                should start with "scan_".
            outputs (List[Dict]): List of outputs from the model.
                Each dict should contain at least the following keys:
                * "y_true": Ground truth results
                * "y_pred": Predicted probabilities.
                * "time_elapsed": Amount of time to load data and run model.
        """
        pass

    def evaluate(self):
        """Evaluate/summarize the performance, after processing all input/output
        pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


def inference_on_dataset(
    model,
    data_loader: Union[DataLoader, Generator],
    evaluator: Union[DatasetEvaluator, Sequence[DatasetEvaluator]],
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (keras.Model):
        generator: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    if isinstance(evaluator, DatasetEvaluator):
        evaluator = [evaluator]

    for e in evaluator:
        e.reset()
    num_warmup = 1
    start_time = time.perf_counter()
    total_compute_time = 0
    total_processing_time = 0
    total_inference_time = 0
    if isinstance(data_loader, Generator):
        iter_loader = data_loader.img_generator_test
        total = data_loader.num_scans(GeneratorState.TESTING)
    else:
        iter_loader = data_loader.inference
        total = data_loader.num_scans()

    start_compute_time = time.perf_counter()
    logger = logging.getLogger(__name__)
    for idx, (input, output) in enumerate(iter_loader(model)):
        total_compute_time += time.perf_counter() - start_compute_time

        start_processing_time = time.perf_counter()
        for e in evaluator:
            e.process([input], [output])
        total_processing_time += time.perf_counter() - start_processing_time

        total_inference_time += output["time_elapsed"]
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_scan = total_compute_time / iters_after_start
        seconds_per_inference = total_inference_time / iters_after_start
        seconds_per_processing = total_processing_time / iters_after_start

        if idx >= num_warmup * 2 or seconds_per_scan > 5:
            total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
            eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
            log_every_n_seconds(
                logging.INFO,
                "Inference done {}/{}. {:.4f} s / scan ({:.4f} inference, "
                "{:.4f} processing). ETA={}".format(
                    idx + 1,
                    total,
                    seconds_per_scan,
                    seconds_per_inference,
                    seconds_per_processing,
                    str(eta),
                ),
                n=5,
            )
        start_compute_time = time.perf_counter()

    eval_start = time.perf_counter()
    logger.info("Begin evaluation...")
    if any([e._config.INFERENCE_ONLY for e in evaluator]):
        results = None
    else:
        results = {e.__class__.__name__: e.evaluate() for e in evaluator}
    total_eval_time = time.perf_counter() - eval_start
    logger.info("Time Elapsed: {:.4f} seconds".format(total_compute_time + total_eval_time))
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream
    # code to handle
    if results is None:
        results = {}
    return results
