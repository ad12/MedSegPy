"""Metrics Processor.

A processor keeps track of task-specific metrics for different classes.
It should not be used to keep track of non-class-specific metrics, such as
runtime.
"""
import inspect
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import Callable, Collection, Dict, Sequence, Union

import numpy as np
import pandas as pd
import tabulate
from medpy.metric import assd

logger = logging.getLogger(__name__)


def flatten_non_category_dims(
    xs: Sequence[np.ndarray], category_dim: int = None
):
    """Flattens all non-category dimensions into a single dimension.

    Args:
        xs (ndarrays): Sequence of ndarrays with the same category dimension.
        category_dim: The dimension/axis corresponding to different categories.
            i.e. `C`. If `None`, behaves like `np.flatten(x)`.

    Returns:
        ndarray: Shape (C, -1) if `category_dim` specified else shape (-1,)
    """
    single_input = isinstance(xs, np.ndarray)
    if single_input:
        xs = [xs]
    if category_dim is not None:
        dims = (xs[0].shape[category_dim], -1)
        xs = (np.moveaxis(x, category_dim, 0).reshape(dims) for x in xs)
    else:
        xs = (x.flatten() for x in xs)

    if single_input:
        xs = list(xs)[0]

    return xs


def rms_cv(y_pred: np.ndarray, y_true: np.ndarray, dim=None):
    """Compute root-mean-squared coefficient of variation.

    This is typically done to compare intra-method variability.
    For example if multiple measurements are taken using the same method.
    However, in many segmentation manuscripts, this is equation is also
    used.

    This quantity is symmetric.

    Args:
        y_pred (ndarray): Measurements from trial 1.
        y_true (ndarray): Measurements from trial 2.
        dim (int, optional): Dimension/axis over which to compute metric.
            If `None`, all dimensions will be reduced.

    Returns:
        ndarray: If `dim=None`, scalar value.
    """
    stds = np.std([y_pred, y_true], axis=0)
    means = np.mean([y_pred, y_true], axis=0)
    cv = stds / means
    return np.sqrt(np.mean(cv ** 2, axis=dim))


def rmse_cv(y_pred: np.ndarray, y_true: np.ndarray, dim=None):
    """Compute root-mean-squared error coefficient of variation.

    This quantity is not symmetric.

    Args:
        y_pred (ndarray): Predicted measurements.
        y_true (ndarray): Ground-truth/baseline measurements.
        dim (int, optional): Dimension/axis over which to compute metric.
            If `None`, all dimensions will be reduced.

    Returns:
        ndarray: If `dim=None`, scalar value.
    """
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=dim))
    means = np.absolute(np.mean(y_true, axis=dim))
    return rmse / means


class Reductions(Enum):
    RMS_CV = 1, "RMS-CV", rms_cv
    RMSE_CV = 2, "RMSE-CV", rmse_cv

    def __new__(cls, value: int, display_name: str, func: Callable):
        """
        Args:
            value (int): Unique integer value.
            patterns (`List[str]`): List of regex patterns that would match the
                hostname on the compute cluster. There can be multiple hostnames
                per compute cluster because of the different nodes.
            save_dir (str): Directory to save data to.
        """
        obj = object.__new__(cls)
        obj._value_ = value

        obj.display_name = display_name
        obj.func = func

        return obj


class Metric(Callable, ABC):
    """Interface for new metrics.

    A metric should be implemented as a callable with explicitly defined
    arguments. In other words, metrics should not have `**kwargs` or `**args`
    options in the `__call__` method.

    While not explicitly constrained to the return type, metrics typically
    return float value(s). The number of values returned corresponds to the
    number of categories.

    * metrics should have different name() for different functionality.
    * `category_dim` duck type if metric can process multiple categories at
        once.

    To compute metrics:

    .. code-block:: python

        metric = Metric()
        results = metric(...)
    """

    def __init__(self, units: str = ""):
        self.units = units

    def name(self):
        return type(self).__name__

    def display_name(self):
        """Name to use for pretty printing and display purposes.
        """
        name = self.name()
        return "{} {}".format(name, self.units) if self.units else name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


class DSC(Metric):
    """Dice score coefficient.
    """

    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        size_i1 = np.count_nonzero(y_pred, -1)
        size_i2 = np.count_nonzero(y_true, -1)
        intersection = np.count_nonzero(y_pred & y_true, -1)

        return 2.0 * intersection / (size_i1 + size_i2)


class VOE(Metric):
    """Volumetric overlap error.
    """

    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        intersection = np.count_nonzero(y_true & y_pred, axis=-1)
        union = np.count_nonzero(y_true | y_pred, axis=-1).astype(np.float)

        return 1 - intersection / union


class CV(Metric):
    """Coefficient of variation.
    """

    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        size_i1 = np.count_nonzero(y_pred, -1)
        size_i2 = np.count_nonzero(y_true, -1)

        std = np.std([size_i1, size_i2], axis=0)
        mean = np.mean([size_i1, size_i2], axis=0)

        return std / mean


class ASSD(Metric):
    """Average symmetric surface distance.
    """

    def __call__(self, y_pred, y_true, spacing=None, connectivity=1):
        # TODO: check why this statement is needed.
        if not connectivity:
            connectivity = 1
        return assd(
            y_pred, y_true, voxelspacing=spacing, connectivity=connectivity
        )


class Precision(Metric):
    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        tp = np.count_nonzero(y_pred & y_true, -1)
        fp = np.count_nonzero(y_pred & ~y_true, -1)

        return tp / (tp + fp)


class Recall(Metric):
    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        tp = np.count_nonzero(y_pred & y_true, -1)
        fn = np.count_nonzero(~y_pred & y_true, -1)

        return tp / (tp + fn)


_BUILT_IN_METRICS = {x.__name__: x for x in Metric.__subclasses__()}


class MetricsManager:
    """A class to manage and compute metrics.

    Metrics will be calculated for the categories specified during
    instantiation. All metrics are assumed to be calculated for those
    categories.

    Metrics are indexed by their string representation as returned by `name()`.
    They are also computed in the order they were added.

    To compute metrics, use this class as a callable. See `__call__` for more
    details.

    Attributes:
        class_names (Sequence[str]): Category names (in order).

    To calculate metrics:

        .. code-block:: python

        manager = MetricsManager(
            category_names=("tumor", "no tumor")
            metrics=(DSC(), VOE())
        )

        for scan_id, x, y_pred, y_true in zip(ids, xs, preds, ground_truths):
            # Compute metrics per scan.
            manager(scan_id, x=x, y_pred=y_pred, y_true=y_true)

    To get number of scans that have been processed:

        .. code-block:: python

        num_scans = len(manager)
    """

    def __init__(
        self,
        class_names: Collection[str],
        metrics: Sequence[Union[Metric, str]] = None,
    ):
        self._scan_data = OrderedDict()
        self.class_names = class_names
        self._metrics: Dict[str, Metric] = OrderedDict()
        self._metric_pairs: Dict[
            (str, str), Sequence[Reductions]
        ] = OrderedDict()  # noqa
        if metrics:
            self.add_metrics(metrics)
        self.category_dim = -1 if len(class_names) > 1 else None
        self.runtimes = []
        self._is_data_stale = False

    def metrics(self):
        """Returns names of current metrics."""
        return self._metrics.keys()

    def add_metrics(self, metrics: Sequence[Union[Metric, str]]):
        """Add metrics to compute.

        Metrics with the same `name()` cannot be added.

        Args:
            metrics (Metric(s)/str(s)): Metrics to compute. `str` values should
                only be used for built-in metrics.

        Raises:
            ValueError: If `metric.name()` already exists. Metrics with the
                same name mean the same computation will be done twice, which
                is not a supported feature.
        """
        if isinstance(metrics, (Metric, str)):
            metrics = [metrics]
        for m in metrics:
            if isinstance(m, str):
                m = _BUILT_IN_METRICS[m]()
            m_name = m.name()
            if m_name in self._metrics:
                raise ValueError("Metric {} already exists".format(m_name))
            self._metrics[m_name] = m

    def register_pairs(
        self,
        pred_metric: str,
        base_metric: str,
        reductions: Union[Reductions, Sequence[Reductions]],
        name: str = None,
    ):
        if pred_metric not in self._metrics:
            raise ValueError("`pred_metric` '{}' not found".format(pred_metric))
        if base_metric not in self._metrics:
            raise ValueError("`base_metric` '{}' not found".format(base_metric))

        key = (pred_metric, base_metric)
        if key in self._metric_pairs:
            raise ValueError("Pair {} already registered".format(key))

        if name is None:
            name = self._metrics[pred_metric].__class__.__name__
        self._metric_pairs[key] = (reductions, name)

    def remove_metrics(self, metrics: Union[str, Sequence[str]]):
        """Remove metrics to compute.

        Args:
            metrics (`str(s)`): Names of metrics to remove.
        """
        if isinstance(metrics, str):
            metrics = [metrics]
        pairs_to_remove = []
        for m in metrics:
            del self._metrics[m]
            pairs_to_remove.extend([x for x in self._metric_pairs if m in x])

        for pair in pairs_to_remove:
            del self._metric_pairs[pair]

    def __call__(
        self,
        scan_id: str,
        x: np.ndarray = None,
        y_pred: np.ndarray = None,
        y_true: np.ndarray = None,
        runtime: float = np.nan,
        **kwargs,
    ) -> str:
        """Compute metrics for a scan.

        Args:
            scan_id (str): The scan/example identifier
            x (ndarray, optional): The input `x` accepted by most metrics.
            y_pred (ndarray, optional): The predicted output.
                For most metrics, should be binarized.
                If computing for multiple classes, last dimension should
                index different categories in the order of `self.class_names`.
            y_true (ndarray, optional): The binarized ground truth output.
                For multiple classes, format like `y_pred`.
            runtime (float, optional): The compute time. If specified, logged
                as an additional metric.

        Returns:
            str: A summary of the results for the scan.
        """
        # Hacky way to define some expected values but still add them
        # to kwargs for future processing.
        for k, v in locals().items():
            if k in ["x", "y_pred", "y_true"] and v is not None:
                kwargs[k] = v

        num_classes = len(self.class_names)
        metrics_names = [m.display_name() for m in self._metrics.values()]
        metrics_data = []
        for metric in self._metrics.values():
            args = inspect.getfullargspec(metric).args

            params = {
                name: kwargs.get(name) for name in args if name not in ["self"]
            }

            if num_classes > 1 and "category_dim" in args:
                params["category_dim"] = self.category_dim
                metrics_data.append(metric(**params))
            else:
                metrics_data.append(
                    [
                        metric(
                            **{
                                k: v[..., c]
                                if k in ["y_pred", "y_true", "x"]
                                else v
                                for k, v in params.items()
                            }
                        )
                        for c in range(num_classes)
                    ]
                )

        metrics_data = pd.DataFrame(
            metrics_data, index=metrics_names, columns=self.class_names
        )
        metrics_data.replace([np.inf, -np.inf], np.nan)
        self.runtimes.append(runtime)

        strs = [
            "{}: {:0.3f}".format(n, v)
            for n, v in zip(metrics_names, np.nanmean(metrics_data, axis=1))
        ]

        self._scan_data[scan_id] = metrics_data
        self._is_data_stale = True

        return ", ".join(strs)

    def _compute_metric_pairs(self, arr):
        pair_names = []
        vals = []
        names = list(self._metrics.keys())
        for (pred_key, base_key), (reds, name) in self._metric_pairs.items():
            pred_idx, base_idx = names.index(pred_key), names.index(base_key)
            preds = arr[:, pred_idx, ...]
            base = arr[:, base_idx, ...]

            vals.extend([r.func(preds, base, dim=0) for r in reds])
            e_names = ["{} {}".format(r.display_name, name) for r in reds]
            pair_names.extend(e_names)

        vals = np.stack(vals, axis=0)
        df = pd.DataFrame(vals, index=pair_names, columns=self.class_names)
        return df

    def scan_summary(self, scan_id, delimiter: str = ", ") -> str:
        """Get summary of results for a scan.

        Args:
            scan_id: Scan id for which to summarize results.
            delimiter (`str`, optional): Delimiter between different metrics.

        Returns:
            str: A summary of metrics for the scan. Values are averaged across
                all categories.
        """
        scan_data = self._scan_data[scan_id]
        avg_data = scan_data.mean(axis=1)

        strs = [
            "{}: {:0.3f}".format(n, avg_data[n])
            for n in avg_data.index.tolist()
        ]

        return delimiter.join(strs)

    def data_frame(self):
        raw_arr = np.stack(
            [np.asarray(x) for x in self._scan_data.values()], axis=0
        )
        arr = np.nanmean(raw_arr, axis=0)
        names = [m.display_name() for m in self._metrics.values()]

        df = pd.DataFrame(arr, index=names, columns=self.class_names)
        if self._metric_pairs:
            df_pairs = self._compute_metric_pairs(raw_arr)
            df = pd.concat([df, df_pairs])

        return df

    def summary(self):
        """Get summary of results overall scans.

        Returns:
            str: Tabulated summary. Rows=metrics. Columns=classes.
        """
        df = self.data_frame()

        return tabulate.tabulate(df, headers=self.class_names) + "\n"

    def data(self):
        """
        TODO: Determine format
        """
        df = pd.concat(self._scan_data.values(), keys=self._scan_data.keys())
        return {"runtimes": self.runtimes, "scan_data": df}

    def __len__(self):
        return len(self._scan_data)

    def __getitem__(self, scan_id):
        return self._scan_data[scan_id]
