"""Stanford abdominal CT Dataset."""
import os
import numpy as np

from medsegpy.evaluation import EVALUATOR_REGISTRY, SemSegEvaluator
from medsegpy.evaluation.metrics import Metric, Reductions, flatten_non_category_dims


class HU(Metric):
    """
    Calculates the mean Hounsfield Unit value.
    """

    def __init__(self, method, units=""):
        super().__init__(units)
        assert method in ["pred", "base"]
        self.method = method

    def _compute_hu(self, x, y, category_dim: int = None):
        y = y.astype(np.bool)
        if category_dim is None:
            return np.mean(x[y])

        num_classes = y.shape[-1]
        assert category_dim == -1
        hu = np.array([np.mean(x[y[..., c]]) for c in range(num_classes)])

        return hu

    def __call__(self, x, y_pred=None, y_true=None, category_dim: int = None):
        if self.method == "pred":
            assert y_pred is not None
            hu = self._compute_hu(x, y_pred, category_dim)
        if self.method == "base":
            assert y_true is not None
            hu = self._compute_hu(x, y_true, category_dim)

        return (hu,)

    def name(self):
        return "HU {}".format(self.method)


class CSA(Metric):
    """
    Calculates the cross-sectional area.
    """

    def __init__(self, method, units=""):
        super().__init__(units)
        assert method in ["pred", "base"]
        self.method = method

    def _compute_area(self, y, pixel_area, category_dim=None):
        y = flatten_non_category_dims(y.astype(np.bool), category_dim)
        return pixel_area * np.count_nonzero(y, -1)

    def __call__(self, y_pred=None, y_true=None, spacing=None, category_dim: int = None):
        pixel_area = np.prod(spacing) if spacing else 1
        if self.method == "pred":
            assert y_pred is not None
            area = self._compute_area(y_pred, pixel_area, category_dim)
        if self.method == "base":
            assert y_true is not None
            area = self._compute_area(y_true, pixel_area, category_dim)

        return (area,)

    def name(self):
        return "Area {}".format(self.method)


class HUDiff(Metric):
    """
    Calculates the absolute difference or the percent error in Hounsfield units
    when compared to the ground truth.
    """

    FULL_NAME = "Hounsfield Unit - Difference"

    def __init__(self, units="", method="absolute"):
        super().__init__(units)
        assert method in ["absolute", "percent"]
        self._method = method

    def __call__(self, y_pred, y_true, x, category_dim: int = None):
        y_true = y_true.astype(np.bool)
        y_pred = y_pred.astype(np.bool)
        if category_dim is None:
            return np.mean(x[y_pred]) - np.mean(x[y_true])

        assert category_dim == -1
        num_classes = y_pred.shape[-1]

        hu_pred = np.array([np.mean(x[y_pred[..., c]]) for c in range(num_classes)])
        hu_true = np.array([np.mean(x[y_true[..., c]]) for c in range(num_classes)])

        vals = hu_pred - hu_true
        if self._method == "percent":
            vals = vals / np.absolute(hu_true) * 100

        return (vals,)

    def name(self):
        symbol = "Abs" if self._method == "absolute" else "%"
        return "HU {}Diff".format(symbol)


class AreaDiff(Metric):
    """
    Calculates the absolute difference or the percent error in cross-sectional area
    when compared to the ground truth.
    """

    def __init__(self, units="", method="absolute"):
        super().__init__(units)
        assert method in ["absolute", "percent"]
        self._method = method

    def __call__(self, y_pred, y_true, spacing=None, category_dim: int = None):
        pixel_area = np.prod(spacing) if spacing else 1
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = flatten_non_category_dims((y_pred, y_true), category_dim)

        size_pred = pixel_area * np.count_nonzero(y_pred, -1)
        size_true = pixel_area * np.count_nonzero(y_true, -1)

        val = size_pred - size_true
        if self._method == "percent":
            val = 100 * val / size_true

        return (val,)

    def name(self):
        symbol = "Abs" if self._method == "absolute" else "%"
        return "Area {}Diff".format(symbol)


@EVALUATOR_REGISTRY.register()
class CTEvaluator(SemSegEvaluator):
    """Evaluate semantic segmentation on CT datasets.

    Includes Hounsfield unit difference and pixel area difference between datasets.
    """

    def _get_metrics(self):
        metrics = list(super()._get_metrics())
        paired_metrics = [
            HU(method="pred"),
            HU(method="base"),
            CSA(method="pred", units="(mm^2)"),
            CSA(method="base", units="(mm^2)"),
        ]
        metrics.extend(
            [
                HUDiff(method="absolute"),
                HUDiff(method="percent"),
                AreaDiff(method="absolute", units="(mm^2)"),
                AreaDiff(method="percent"),
            ]
        )
        metrics.extend(paired_metrics)

        self._metric_pairs = [
            (p.name(), b.name()) for p, b in zip(paired_metrics[::2], paired_metrics[1::2])
        ]
        return metrics

    def reset(self):
        super().reset()
        for pred_key, base_key in self._metric_pairs:
            self._metrics_manager.register_pairs(
                pred_key, base_key, (Reductions.RMS_CV, Reductions.RMSE_CV)
            )

    def process(self, inputs, outputs):
        if not self.spacing:
            assert all("scan_spacing" in x for x in inputs)
        super().process(inputs, outputs)


@EVALUATOR_REGISTRY.register()
class CTEvaluatorTTPP(CTEvaluator):
    """CT Evaluator with test-time post-processing.
    """

    def __init__(
        self,
        dataset_name: str,
        cfg,
        output_folder: str = None,
        save_raw_data: bool = False,
        stream_evaluation: bool = True,
    ):
        super().__init__(
            dataset_name,
            cfg,
            os.path.join(cfg.OUTPUT_DIR, "test_results_pp-muscle-imat"),
            save_raw_data,
            stream_evaluation,
        )

    def _postprocess_labels(self, x, labels):
        labels = labels.copy()
        categories = self._categories
        muscle_idx = categories.index("muscle")
        imat_idx = categories.index("intramuscular fat")

        muscle_mask = labels[..., muscle_idx]
        imat_mask = labels[..., imat_idx]

        imat_mask[muscle_mask.astype(np.bool) & (x < -30)] = 1
        muscle_mask[x < -30] = 0

        labels[..., muscle_idx] = muscle_mask
        labels[..., imat_idx] = imat_mask

        return labels

    def process(self, inputs, outputs):
        if not self.spacing:
            assert all("scan_spacing" in x for x in inputs)

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
                raise ValueError("output activation {} not supported".format(output_activation))

            # background is always excluded from analysis
            if includes_bg:
                y_true = output["y_true"][..., 1:]
                y_pred = output["y_pred"][..., 1:]
                labels = labels[..., 1:]
                output["y_true"] = y_true
                output["y_pred"] = y_pred

            # Post process labels to switch muscle/imat labels at hu threshold of -40
            labels = self._postprocess_labels(input["x"], labels)

            time_elapsed = output["time_elapsed"]
            if self.stream_evaluation:
                self.eval_single_scan(input, output, labels, time_elapsed)
            else:
                self._predictions.append((input, output, labels, time_elapsed))
