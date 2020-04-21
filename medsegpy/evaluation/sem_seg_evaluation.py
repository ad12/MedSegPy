import logging
import os
import time

from fvcore.common.file_io import PathManager
import h5py
import numpy as np
import seaborn as sns

from medsegpy.config import Config
from medsegpy.data import MetadataCatalog
from medsegpy.utils.metric_utils import MetricsManager
from medsegpy.utils.metric_utils import SegMetric

from .build import EVALUATOR_REGISTRY
from .evaluator import DatasetEvaluator


def get_stats_string(mw: MetricsManager, testing_time):
    """
    Return string detailing statistics
    :param mw:
    :param skipped_count:
    :param testing_time:
    :return:
    """
    seg_metrics_processor = mw.seg_metrics_processor
    inference_runtimes = np.asarray(mw.runtimes)

    s = "============ Overall Summary ============\n"
    s += "Time elapsed: %0.1f seconds.\n" % testing_time
    s += "%s\n" % seg_metrics_processor.summary()
    s += (
        "Inference time (Mean +/- Std. Dev.): "
        "{:0.2f} +/- {:0.2f} seconds.\n".format(
            np.mean(inference_runtimes), np.std(inference_runtimes)
        )
    )
    return s


@EVALUATOR_REGISTRY.register()
class SemSegEvaluator(DatasetEvaluator):
    """
    Evaluate semantic segmentation
    """

    def __init__(
        self,
        dataset_name,
        cfg: Config,
        output_folder=None,
        save_raw_data: bool = False,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            cfg:
            output_folder (str): an output directory to dump results.
            save_raw_data (:obj:`bool`, optional): Save recon, labels, ground
                truth masks to h5 file.
        """
        self._config = cfg
        self._dataset_name = dataset_name
        self._output_folder = output_folder \
            if output_folder else os.path.join(cfg.OUTPUT_DIR, "test_results")
        self._num_classes = cfg.get_num_classes()
        self._ignore_label = 0

        self._logger = logging.getLogger(__name__)

        meta = MetadataCatalog.get(dataset_name)
        self._meta = meta
        cat_ids = cfg.TISSUES
        contiguous_id_map = meta.get("category_id_to_contiguous_id")
        contiguous_ids = [contiguous_id_map[x] for x in cat_ids]

        categories = meta.get("categories")
        categories = [categories[c_id] for c_id in contiguous_ids]
        category_colors = meta.get("category_colors")
        if category_colors:
            category_colors = [category_colors[c_id] for c_id in contiguous_ids]
        else:
            category_colors = sns.color_palette("bright")
        self._categories = categories
        self._category_colors = category_colors
        self._metrics = [SegMetric[m] for m in cfg.TEST_METRICS]

        self._metrics_manager = None
        self._predictions = None
        self._voxel_spacing = meta.get("voxel_spacing", None)

        self._save_raw_data = save_raw_data

        self._output_activation = cfg.LOSS[1]
        self._output_includes_background = cfg.INCLUDE_BACKGROUND

    def reset(self):
        self._metrics_manager = MetricsManager(
            class_names=self._categories,
            metrics=self._metrics
        )
        self._predictions = []

    def process(self, inputs, outputs, times_elapsed):
        """
        Args:
            inputs (List[Dict[str, Any]]): The inputs to the model.
                Each dict corresponds to a scan id, input volume/image, and
                (Dx)HxWxC ground truth mask.
            outputs (List[ndarray]): The outputs of the model.
                Each ndarray corresponds to the (Dx)HxWxC probability tensor
                output by semantic segmentation models.
            times_elapsed (List[float]): The segmentation times per input.
        """
        output_activation = self._output_activation
        includes_bg = self._output_includes_background

        for input, output, time_elapsed in zip(inputs, outputs, times_elapsed):
            if output_activation == "sigmoid":
                labels = (output > 0.5).astype(np.uint8)
            elif output_activation == "softmax":
                labels = np.zeros_like(output, dtype=np.uint8)
                l_argmax = np.argmax(output, axis=-1)
                for c in range(labels.shape[-1]):
                    labels[l_argmax == c, c] = 1
                labels = labels.astype(np.uint)
            else:
                raise ValueError("output activation {} not supported".format(
                    output_activation
                ))

            # background is always excluded from analysis
            if includes_bg:
                y_true = input["y_true"][..., 1:]
                output = output[..., 1:]
                labels = labels[..., 1:]
                if y_true.ndim == 3:
                    y_true = y_true[..., np.newaxis]
                    output = output[..., np.newaxis]
                    labels = labels[..., np.newaxis]
                input["y_true"] = y_true

            self._predictions.append((input, output, labels, time_elapsed))

    def evaluate(self):
        """Evaluates popular medical segmentation metrics specified in config.

        * Evaluate on popular medical segmentation metrics. For supported
          segmentation metrics, see :class:`MetricsManager`.
        * Save overlay images.
        * Save probability predictions.

        Note, by default coefficient of variation (CV) is calculated as a
        root-mean-squared quantity rather than mean.
        """
        metrics_manager = self._metrics_manager
        voxel_spacing = self._voxel_spacing
        logger = self._logger
        save_raw_data = self._save_raw_data
        output_dir = self._output_folder
        PathManager.mkdirs(output_dir)

        scan_cnt = 0
        results_str = ""
        start = time.time()
        for input, output, labels, time_elapsed in self._predictions:
            scan_cnt += 1
            scan_id = input["scan_id"]
            y_true = input["y_true"]

            summary = metrics_manager.analyze(
                scan_id,
                np.transpose(y_true, axes=[1, 2, 0, 3]),
                np.transpose(labels, axes=[1, 2, 0, 3]),
                voxel_spacing=voxel_spacing,
                runtime=time_elapsed,
            )

            logger_info_str = (
                "Scan #{:03d} (name = {}, {:0.2f}s) = {}".format(
                    scan_cnt, scan_id, time_elapsed, summary
                )
            )
            results_str = results_str + logger_info_str + "\n"
            logger.info(logger_info_str)

            if output_dir and save_raw_data:
                save_name = "{}/{}.pred".format(output_dir, scan_id)
                with h5py.File(save_name, "w") as h5f:
                    h5f.create_dataset("recon", data=output)
                    h5f.create_dataset("labels", data=labels)
                    h5f.create_dataset("gt", data=y_true)

            # TODO: Save overlay images as tiff/jpeg.

        end = time.time()

        stats_string = get_stats_string(metrics_manager, end - start)
        logger.info('--' * 20)
        logger.info(stats_string)
        logger.info('--' * 20)
        if output_dir:
            test_results_summary_path = os.path.join(output_dir, "results.txt")

            # Write details to test file
            with open(test_results_summary_path, 'w+') as f:
                f.write('Results generated on %s\n' % time.strftime('%X %x %Z'))
                f.write('Weights Loaded: %s\n' % os.path.basename(self._config.TEST_WEIGHT_PATH))
                f.write('Voxel Spacing: %s\n' % str(voxel_spacing))
                f.write('--' * 20)
                f.write('\n')
                f.write(results_str)
                f.write('--' * 20)
                f.write('\n')
                f.write(stats_string)

        # TODO: Convert segmentation metrics to valid results matrix.
        return {}
