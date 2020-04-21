"""Dataset evaluator.

Adopted from Facebook's detectron2.
https://github.com/facebookresearch/detectron2
"""
import datetime
import logging
import time

from medsegpy.utils.logger import log_every_n_seconds
from medsegpy.data.im_gens import Generator
from medsegpy.data.im_gens import GeneratorState


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, inputs, outputs, time_elapsed):
        """
        Process an input/output pair.

        Args:
            scan_id: the scan id corresponding to the input/output
            input: the input that's used to call the model.
            output: the return value of `model(input)`
        """

        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

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
    generator: Generator,
    evaluator: DatasetEvaluator,
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    The model will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        generator: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use
            :class:`DatasetEvaluators([])` if you only want to benchmark, but
            don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    evaluator.reset()
    num_warmup = 2
    start_time = time.perf_counter()
    total_compute_time = 0
    total = generator.num_scans(GeneratorState.TESTING)

    for idx, (x_test, y_test, recon, fname, time_elapsed) in enumerate(generator.img_generator_test(model)):
        input = {"scan_id": fname, "y_true": y_test, "scan": x_test}
        evaluator.process(
            [input],
            [recon],
            [time_elapsed],
        )
        iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
        seconds_per_scan = total_compute_time / iters_after_start
        if idx >= num_warmup * 2 or seconds_per_scan > 5:
            total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
            eta = datetime.timedelta(
                seconds=int(total_seconds_per_img * (total - idx - 1)))
            log_every_n_seconds(
                logging.INFO,
                "Inference done {}/{}. {:.4f} s / scan. ETA={}".format(
                    idx + 1, total, seconds_per_scan, str(eta)
                ),
                n=5,
            )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results
