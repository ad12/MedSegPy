import logging
import os
import time
from typing import Sequence, Union

import h5py
import numpy as np
from fvcore.common.file_io import PathManager

from medsegpy.config import Config
from medsegpy.data import MetadataCatalog

from .build import EVALUATOR_REGISTRY
from .evaluator import DatasetEvaluator
from .metrics import MetricsManager, Metric


def get_stats_string(manager: MetricsManager):
    """Returns formatted metrics manager summary string.

    Args:
        manager (MetricsManager): The manager whose results to format.

    Returns:
        str: A formatted string detailing manager results.
    """
    inference_runtimes = np.asarray(manager.runtimes)

    s = "============ Overall Summary ============\n"
    s += "%s\n" % manager.summary()
    s += (
        "Inference time (Mean +/- Std. Dev.): "
        "{:0.2f} +/- {:0.2f} seconds.\n".format(
            np.mean(inference_runtimes), np.std(inference_runtimes)
        )
    )
    return s


@EVALUATOR_REGISTRY.register()
class SemSegEvaluator(DatasetEvaluator):
    """Evaluator for semantic segmentation-related tasks.
    """
    def __init__(
        self,
        dataset_name: str,
        cfg: Config,
        output_folder: str=None,
        save_raw_data: bool = False,
        stream_evaluation: bool = True,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg:
            output_folder (str): an output directory to dump results.
            save_raw_data (:obj:`bool`, optional): Save probs, labels, ground
                truth masks to h5 file.
            stream_evaluation (:obj:`bool`, optional): If `True`, evaluates
                data as it comes in to avoid holding too many objects in memory.
        """
        self._config = cfg
        self._dataset_name = dataset_name
        self._output_folder = (
            output_folder
            if output_folder
            else os.path.join(cfg.OUTPUT_DIR, "test_results")
        )
        PathManager.mkdirs(self._output_folder)
        self._num_classes = cfg.get_num_classes()
        self._ignore_label = 0

        self._logger = logging.getLogger(__name__)

        meta = MetadataCatalog.get(dataset_name)
        self._meta = meta
        cat_ids = cfg.CATEGORIES
        contiguous_id_map = meta.get("category_id_to_contiguous_id")
        contiguous_ids = [
            contiguous_id_map[tuple(x) if isinstance(x, list) else x]
            for x in cat_ids
        ]
        categories = meta.get("categories")
        categories = [categories[c_id] for c_id in contiguous_ids]
        self.spacing = meta.get("spacing", None)

        self._categories = categories

        self._metrics_manager = None
        self._predictions = None
        self._scan_cnt = 0
        self._results_str = ""

        self._save_raw_data = save_raw_data
        self.stream_evaluation = stream_evaluation

        self._output_activation = cfg.LOSS[1]
        self._output_includes_background = cfg.INCLUDE_BACKGROUND

    def _get_metrics(self) -> Sequence[Union[str, Metric]]:
        """"""
        return self._config.TEST_METRICS

    def reset(self):
        self._metrics_manager = MetricsManager(
            class_names=self._categories, metrics=self._get_metrics()
        )
        self._predictions = []
        self._scan_cnt = 0
        self._results_str = ""

    def process(self, inputs, outputs):
        """
        See :class:`DatasetEvaluator` in evaluator.py
        for argument details.
        """
        output_activation = self._output_activation
        includes_bg = self._output_includes_background

        for input, output in zip(inputs, outputs):
            y_pred = output["y_pred"]

            if output_activation == "sigmoid":
                labels = (y_pred > 0.5).astype(np.uint8)
            elif output_activation == "softmax":
                labels = np.zeros_like(y_pred, dtype=np.uint8)
                l_argmax = np.argmax(y_pred, axis=-1)
                for c in range(labels.shape[-1]):
                    labels[l_argmax == c, c] = 1
                labels = labels.astype(np.uint)
            else:
                raise ValueError(
                    "output activation {} not supported".format(
                        output_activation
                    )
                )

            # background is always excluded from analysis
            if includes_bg:
                y_true = output["y_true"][..., 1:]
                y_pred = output["y_pred"][..., 1:]
                labels = labels[..., 1:]
                if y_true.ndim == 3:
                    y_true = y_true[..., np.newaxis]
                    y_pred = y_pred[..., np.newaxis]
                    labels = labels[..., np.newaxis]
                output["y_true"] = y_true
                output["y_pred"] = y_pred

            time_elapsed = output["time_elapsed"]
            if self.stream_evaluation:
                self.eval_single_scan(input, output, labels, time_elapsed)
            else:
                self._predictions.append((input, output, labels, time_elapsed))

    def eval_single_scan(self, input, output, labels, time_elapsed):
        metrics_manager = self._metrics_manager
        spacing = input["scan_spacing"] if "scan_spacing" in input else self.spacing
        logger = self._logger
        save_raw_data = self._save_raw_data
        output_dir = self._output_folder

        self._scan_cnt += 1
        scan_cnt = self._scan_cnt
        scan_id = input["scan_id"]
        y_true: np.ndarray = output["y_true"]
        x: np.ndarray = np.squeeze(input["x"])

        metrics_kwargs = {"spacing": spacing} if spacing is not None else {}
        summary = metrics_manager(
            scan_id,
            y_true=np.squeeze(np.transpose(y_true, axes=[1, 2, 0, 3])),
            y_pred=np.squeeze(np.transpose(labels, axes=[1, 2, 0, 3])),
            x=x,
            runtime=time_elapsed,
            **metrics_kwargs,
        )

        logger_info_str = "Scan #{:03d} (name = {}, {:0.2f}s) = {}".format(
            scan_cnt, scan_id, time_elapsed, summary
        )
        self._results_str = self._results_str + logger_info_str + "\n"
        logger.info(logger_info_str)

        if output_dir and save_raw_data:
            save_name = "{}/{}.pred".format(output_dir, scan_id)
            with h5py.File(save_name, "w") as h5f:
                h5f.create_dataset("probs", data=output["y_pred"])
                h5f.create_dataset("labels", data=labels)

    def evaluate(self):
        """Evaluates popular medical segmentation metrics specified in config.

        * Evaluate on popular medical segmentation metrics. For supported
          segmentation metrics, see :class:`MetricsManager`.
        * Save overlay images.
        * Save probability predictions.

        Note, by default coefficient of variation (CV) is calculated as a
        root-mean-squared quantity rather than mean.
        """
        output_dir = self._output_folder
        spacing = self.spacing
        logger = self._logger

        if self._predictions:
            for input, output, labels, time_elapsed in self._predictions:
                self.eval_single_scan(input, output, labels, time_elapsed)

        results_str = self._results_str
        stats_string = get_stats_string(self._metrics_manager)
        logger.info("--" * 20)
        logger.info("\n" + stats_string)
        logger.info("--" * 20)
        if output_dir:
            test_results_summary_path = os.path.join(output_dir, "results.txt")

            # Write details to test file
            with open(test_results_summary_path, "w+") as f:
                f.write("Results generated on %s\n" % time.strftime("%X %x %Z"))
                f.write(
                    "Weights Loaded: %s\n"
                    % os.path.basename(self._config.TEST_WEIGHT_PATH)
                )

                f.write("--" * 20)
                f.write("\n")
                f.write(results_str)
                f.write("--" * 20)
                f.write("\n")
                f.write(stats_string)

        # TODO: Convert segmentation metrics to valid results matrix.
        return {}
