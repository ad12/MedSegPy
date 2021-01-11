"""Test built-in metrics.

Most metrics have 2 built-in methods:

    1. `test_consistency`: Test that output of metric is consistent with
        existing implementations (typically from `medpy`)
    2. `test_multiclass`: If the metric supports multi-class, it should have
        a argument named `category_dim` in its callable arguments. The
        results should also be the same as processing each class one by one.

While many of these methods are repeated across classes, we choose not to use
inheritance. This is because each unittest should encapsulate functionality
specific to the class. Instead, we provide helper methods that handle
overlapping functionality:

    * `_binary_arr`: Creates a random binary array of shape `shape`
    * `_is_consistent_medpy`: Helper method for `test_consistency` with
      metrics from the `medpy` library.
    * `_test_multiclass`: Helper method to verify that the multi-class
       implementation works as expected.
"""

import inspect
import unittest

import numpy as np
from medpy.metric import assd, dc, precision, recall

from medsegpy.evaluation.metrics import ASSD, CV, DSC, VOE, Metric, Precision, Recall

_SHAPE = (300, 300, 40)
_NUM_CLASSES = 4
_MULTI_CLASS_SHAPE = _SHAPE + (_NUM_CLASSES,)
_CATEGORY_DIM = -1


def _binary_arr(shape):
    """Make binary array.

    Args:
        shape: Shape of array to make.

    Returns:
        ndarray: A binary array.
    """
    return (np.random.random_sample(shape) > 0.5).astype(np.uint8)


def _is_consistent_medpy(metric: Metric, func, supports_multiclass: bool):
    """Test consistency between metric and medpy implementation.

    Args:
        metric (Metric): A metric
        func (function): MedPy equivalent of metric. Should accept binary input
            in order `result`, `reference`. If you would like to load other
            variables in, provide a function with partial arguments.
        supports_multiclass (bool): If `metric` is expected to support
            multiclass implementation. If `True`, check for duck-typing
            of `category_dim` argument.
    """
    # Single class
    y_true = _binary_arr(_SHAPE)
    y_pred = _binary_arr(_SHAPE)
    args = {"y_pred": y_pred, "y_true": y_true}
    assert np.allclose(func(y_pred, y_true), metric(**args))

    # Multiple classes
    num_classes = _NUM_CLASSES
    y_true = _binary_arr(_MULTI_CLASS_SHAPE)
    y_pred = _binary_arr(_MULTI_CLASS_SHAPE)

    if supports_multiclass:
        category_dim = _CATEGORY_DIM
        args = {"y_pred": y_pred, "y_true": y_true}
        metrics_vals = metric(category_dim=category_dim, **args)
    else:
        metrics_vals = np.asarray(
            [metric(y_pred[..., c], y_true[..., c]) for c in range(num_classes)]
        )

    base_vals = np.asarray(
        [func(result=y_pred[..., c], reference=y_true[..., c]) for c in range(num_classes)]
    )

    assert np.allclose(base_vals, metrics_vals)


def _test_multiclass(metric: Metric):
    """Test multi-class functionality in :class:`Metric`.

    Verify the following:
        1. `metric(...)` is duck-typed with `category_dim` argument
        2. `category_dim` defaults to `None`
        3. `category_dim` can be any dimension

    Args:
        metric:

    Returns:

    """
    arg_names = inspect.getfullargspec(metric).args
    assert (
        "category_dim" in arg_names
    ), "Metrics supporting multiple categories must have 'category_dim' arg"

    idx = arg_names.index("category_dim") - len(arg_names)
    defaults = inspect.getfullargspec(metric).defaults
    assert defaults[idx] is None, "Default value for 'category_dim' should be `None`."

    # Standard category_dim=-1
    shape = (100, 100, 100, 4)
    category_dim = -1
    num_classes = shape[category_dim]
    y_true = _binary_arr(shape)
    y_pred = _binary_arr(shape)

    args = {"y_pred": y_pred, "y_true": y_true}
    metrics_vals = metric(category_dim=category_dim, **args)
    base_vals = np.asarray([metric(y_pred[..., c], y_true[..., c]) for c in range(num_classes)])
    assert np.allclose(base_vals, metrics_vals), "category_dim={}".format(category_dim)

    # Standard category_dim=1
    shape = (100, 6, 100, 100)
    category_dim = 1
    num_classes = shape[category_dim]
    y_true = _binary_arr(shape)
    y_pred = _binary_arr(shape)

    args = {"y_pred": y_pred, "y_true": y_true}
    metrics_vals = metric(category_dim=category_dim, **args)
    base_vals = np.asarray(
        [metric(y_pred[:, c, ...], y_true[:, c, ...]) for c in range(num_classes)]
    )
    assert np.allclose(base_vals, metrics_vals), "category_dim={}".format(category_dim)


class TestDSC(unittest.TestCase):
    def test_consistency(self):
        _is_consistent_medpy(DSC(), dc, True)

    def test_multiclass(self):
        _test_multiclass(DSC())


class TestVOE(unittest.TestCase):
    @staticmethod
    def _expected_voe(result, reference):
        y_true = reference.flatten()
        y_pred = result.flatten()

        y_true_bool = np.asarray(y_true, dtype=np.bool)
        y_pred_bool = np.asarray(y_pred, dtype=np.bool)
        TP = np.sum(y_true_bool * y_pred_bool, axis=-1)
        FP = np.sum(~y_true_bool * y_pred_bool, axis=-1)
        FN = np.sum(y_true_bool * ~y_pred_bool, axis=-1)

        voe = 1 - (TP) / (TP + FP + FN)

        return voe

    def test_consistency(self):
        _is_consistent_medpy(VOE(), self._expected_voe, True)

    def test_multiclass(self):
        _test_multiclass(VOE())


class TestCV(unittest.TestCase):
    @staticmethod
    def _expected_cv(result, reference):
        y_true = np.squeeze(reference)
        y_pred = np.squeeze(result)

        cv = np.std([np.sum(y_true), np.sum(y_pred)]) / np.mean([np.sum(y_true), np.sum(y_pred)])
        return cv

    def test_consistency(self):
        _is_consistent_medpy(CV(), self._expected_cv, True)

    def test_multiclass(self):
        _test_multiclass(CV())


class TestASSD(unittest.TestCase):
    def test_consistency(self):
        _is_consistent_medpy(ASSD(), assd, False)


class TestPrecision(unittest.TestCase):
    def test_consistency(self):
        _is_consistent_medpy(Precision(), precision, True)

    def test_multiclass(self):
        _test_multiclass(Precision())


class TestRecall(unittest.TestCase):
    def test_consistency(self):
        _is_consistent_medpy(Recall(), recall, True)

    def test_multiclass(self):
        _test_multiclass(Recall())


if "__name__" == "__main__":
    unittest.main()
