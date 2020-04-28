"""Metrics Processor.

A processor keeps track of task-specific metrics for different classes.
It should not be used to keep track of non-class-specific metrics, such as
runtime.
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import inspect
import logging
from enum import Enum
from typing import Collection, Callable, Sequence, Union, Dict
import numpy as np
import pandas as pd
import scipy.stats as spstats
import tabulate
from medpy.metric import assd

logger = logging.getLogger(__name__)


def cv(y_pred: np.ndarray, y_true: np.ndarray, ):
    """Coefficient of Variation.

    Args:
        y_true (ndarray): Binary ground truth
        y_pred (ndarray): Binary prediction

    Returns:
        float
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    cv = np.std([np.sum(y_true), np.sum(y_pred)]) / np.mean(
        [np.sum(y_true), np.sum(y_pred)]
    )
    return cv


def volumetric_overlap_error(y_pred, y_true):
    """Volumetric overlap error

    Args:
        y_true (ndarray): Binary ground truth
        y_pred (ndarray): Binary prediction

    Returns:
        float
    """

    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    y_true_bool = np.asarray(y_true, dtype=np.bool)
    y_pred_bool = np.asarray(y_pred, dtype=np.bool)
    TP = np.sum(y_true_bool * y_pred_bool, axis=-1)
    FP = np.sum(~y_true_bool * y_pred_bool, axis=-1)
    FN = np.sum(y_true_bool * ~y_pred_bool, axis=-1)

    mu = 1e-07

    voe = 1 - (TP + mu) / (TP + FP + FN + mu)

    return voe


def sem(x, **kwargs):
    """Standard error of the mean.

    Returns:
        float
    """
    args = {"axis": 0, "ddof": 0}
    args.update(**kwargs)
    return spstats.sem(x, **args)


class MetricOperation(Enum):
    MEAN = 1, lambda x, **kwargs: np.mean(x, **kwargs)
    MEDIAN = 2, lambda x, **kwargs: np.median(x, **kwargs)
    RMS = 3, lambda x, **kwargs: np.sqrt(np.mean(x ** 2, **kwargs))

    def __new__(cls, keycode, func):
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.func = func
        return obj

    def __call__(self, x: np.ndarray, **kwargs):
        return self.func(np.asarray(x), **kwargs)


class MetricError(Enum):
    STANDARD_DEVIATION = 1, lambda x, **kwargs: np.std(x, **kwargs)
    STANDARD_ERROR = 2, lambda x, **kwargs: sem(x, **kwargs)

    def __new__(cls, keycode, func):
        obj = object.__new__(cls)
        obj._value_ = keycode
        obj.func = func
        return obj

    def __call__(self, x: np.ndarray, **kwargs):
        return self.func(np.asarray(x), **kwargs)


class Metric(Callable, ABC):
    NAME = ""
    FULL_NAME = ""

    def __init__(self, unit: str=""):
        assert self.FULL_NAME
        self.unit = unit  # can be changed at runtime.

    def name(self):
        return self.NAME if self.NAME else type(self).__name__

    def full_name(self):
        return self.FULL_NAME

    def display_name(self):
        """If `self.unit` is defined, appends it to the name."""
        return "{} {}".format(self.name, self.unit) if self.unit else self.name

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass


def _flatten_non_category_dims(xs: Sequence[np.ndarray], category_dim: int=None):
    """Assumes all arrays are of the same shape."""
    if category_dim is not None:
        dims = (xs[0].shape[category_dim], -1)
        xs = (np.moveaxis(x, category_dim, 0).reshape(dims) for x in xs)
    else:
        xs = (x.flatten() for x in xs)

    return xs


class DSC(Metric):
    FULL_NAME = "Dice Score Coefficient"

    def __call__(self, y_pred, y_true, category_dim: int=None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = _flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        size_i1 = np.count_nonzero(y_pred, -1)
        size_i2 = np.count_nonzero(y_true, -1)
        intersection = np.count_nonzero(y_pred & y_true, -1)

        return 2. * intersection / (size_i1 + size_i2)


class VOE(Metric):
    FULL_NAME = "Volumetric Overlap Error"

    def __call__(self, y_pred, y_true, category_dim: int=None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = _flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        intersection = np.count_nonzero(y_true & y_pred, axis=-1)
        union = np.count_nonzero(y_true | y_pred, axis=-1).astype(np.float)

        return 1 - intersection / union


class CV(Metric):
    FULL_NAME = "Coefficient of Variation"

    def __call__(self, y_pred, y_true, category_dim: int=None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = _flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        size_i1 = np.count_nonzero(y_pred, -1)
        size_i2 = np.count_nonzero(y_true, -1)

        std = np.std([size_i1, size_i2], axis=0)
        mean = np.mean([size_i1, size_i2], axis=0)

        return std / mean


class ASSD(Metric):
    FULL_NAME = "Average Symmetric Surface Distance"

    def __call__(self, y_pred, y_true, spacing=None, connectivity=1):
        return assd(
            y_pred, y_true, voxelspacing=spacing, connectivity=connectivity
        )


class Precision(Metric):
    FULL_NAME = "Precision"

    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = _flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        tp = np.count_nonzero(y_pred & y_true, -1)
        fp = np.count_nonzero(y_pred & ~y_true, -1)

        return tp / (tp + fp)


class Recall(Metric):
    FULL_NAME = "Recall"

    def __call__(self, y_pred, y_true, category_dim: int = None):
        y_pred = y_pred.astype(np.bool)
        y_true = y_true.astype(np.bool)
        y_pred, y_true = _flatten_non_category_dims(
            (y_pred, y_true), category_dim
        )

        tp = np.count_nonzero(y_pred & y_true, -1)
        fn = np.count_nonzero(~y_pred & y_true, -1)

        return tp / (tp + fn)


_BUILT_IN_METRICS = {
    x.NAME: x for x in Metric.__subclasses__()
}


class MetricsManager:
    """A class to manage different metrics to use.

    A metric is defined by it's name and a callable to execute when computing
    that metric. We assume that all metrics return float values. It is
    represented as a dictionary element with the following fields:

        * "name": The name of the metric.
        * "func": The callable to execute when computing this metric.
        * "args": The ordered list of arguments names for the function.
        * "defaults": An ordered dict of argument names to their default values.
            If no default arguments are available, an empty dictionary.
        * "full_name" (optional): The full name of the metric (optional).
        * "unit": Unit of the metric.

    A MetricsManager object keeps track of metrics that are to be computed for
    a specific
    """
    def __init__(
        self,
        class_names: Collection[str],
        metrics: Sequence[Union[Metric, str]] = None,
    ):
        self._scan_data = OrderedDict()
        self.class_names = class_names
        self._metrics: Dict[str, Metric] = OrderedDict()
        self.add_metrics(metrics)
        self.category_dim = -1 if len(class_names) > 1 else None
        self.runtimes = []
        self._is_data_stale = False

    def metrics(self):
        return self._metrics.keys()

    def add_metrics(self, metrics: Sequence[Union[Metric, str]]):
        if isinstance(metrics, (Metric, str)):
            metrics = [metrics]
        for m in metrics:
            if isinstance(m, str):
                m = _BUILT_IN_METRICS[m]()
            m_name = m.name()
            if m_name in self._metrics:
                raise ValueError("Metric {} already exists".format(m_name))
            self._metrics[m_name] = m

    def remove_metrics(self, metrics: Union[str, Sequence[str]]):
        if isinstance(metrics, str):
            metrics = [metrics]
        for m in metrics:
            del self._metrics[m]

    def __call__(
        self,
        scan_id: str,
        runtime: float = np.nan,
        **kwargs,
    ) -> str:
        num_classes = len(self.class_names)
        metrics_names = [m.display_name() for m in self._metrics.values()]
        metrics_data = []
        for metric in self._metrics.values():
            args = inspect.getfullargspec(metric).args
            params = {name: kwargs.get(name) for name in args}

            if num_classes > 1 and "category_dim" in args:
                params["category_dim"] = self.category_dim
                metrics_data.append(**params)
            else:
                metrics_data.append([
                    metric(**{
                        k: v[..., c] if k in ["y_pred", "y_true", "x"] else v
                        for k, v in params.items()
                    })
                    for c in range(num_classes)
                ])

        metrics_data = pd.DataFrame(
            metrics_data, index=metrics_names, columns=self.class_names,
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

    def scan_summary(self, scan_id) -> str:
        scan_data = self._scan_data[scan_id]
        avg_data = scan_data.mean(axis=1)

        strs = [
            "{}: {:0.3f}".format(n, avg_data[n])
            for n in avg_data.index.tolist()
        ]

        return ", ".join(strs)

    def summary(self):
        arr = np.stack(
            [np.asarray(x) for x in self._scan_data.values()], axis=0
        )
        arr = np.nanmean(arr, axis=0)
        names = [m.display_name() for m in self._metrics.values()]

        df = pd.DataFrame(arr, index=names, columns=self.class_names)
        return tabulate.tabulate(df, headers=self.class_names) + "\n"

    def data(self):
        df = pd.concat(self._scan_data.values(), keys=self._scan_data.keys())
        return {
            "scan_ids": list(self._scan_data.keys()),
            "runtimes": self.runtimes,
            "scan_data": df,
        }

    def __len__(self):
        return len(self._scan_data)

    def __getitem__(self, scan_id):
        return self._scan_data[scan_id]
