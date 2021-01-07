import unittest

import keras.backend as K
import numpy as np

from medsegpy.loss.classification import DiceLoss
from medsegpy.loss.utils import to_numpy
from medsegpy.losses import dice_loss, avg_dice_loss, multi_class_dice_loss
from medsegpy.utils import env

GT = np.asarray([
    [[0,1,1,0,0],[1,0,0,0,0],[0,0,0,1,1]],
    [[0,1,1,1,0],[1,1,0,0,0],[1,0,0,0,0]],
    [[1,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
])
PRED = np.asarray([
    [[0.1,0.9,0.9,0.1,0.1],[0.9,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.9,0.9]],
    [[0.1,0.9,0.9,0.1,0.1],[0.9,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.9,0.9]],
    [[1,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
])
EPS = 1e-8
    

class TestDiceLoss(unittest.TestCase):
    def test_flatten(self):
        gt = GT
        pred = PRED
        eps = EPS

        loss = DiceLoss(flatten=False, eps=eps, reduction=None)
        dim = 1
        expected = 1 - (2 * (np.sum(gt * pred, axis=dim) + eps) / (np.sum(gt, axis=dim) + np.sum(pred, axis=dim) + eps))
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert val.shape == (3, 5), val.shape
        assert np.allclose(val, expected)

        loss = DiceLoss(flatten="batch", eps=eps, reduction=None)
        dim = (0, 1)
        expected = 1 - (2 * (np.sum(gt * pred, axis=dim) + eps) / (np.sum(gt, axis=dim) + np.sum(pred, axis=dim) + eps))
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert val.shape == (5,), val.shape
        assert np.allclose(val, expected)

        loss = DiceLoss(flatten="channel", eps=eps, reduction=None)
        dim = (1, 2)
        expected = 1 - (2 * (np.sum(gt * pred, axis=dim) + eps) / (np.sum(gt, axis=dim) + np.sum(pred, axis=dim) + eps))
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert val.shape == (3,), val.shape
        assert np.allclose(val, expected)

        loss = DiceLoss(flatten=True, eps=eps, reduction=None)
        dim = (0, 1, 2)
        expected = 1 - (2 * (np.sum(gt * pred, axis=dim) + eps) / (np.sum(gt, axis=dim) + np.sum(pred, axis=dim) + eps))
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert np.isscalar(val)
        assert np.allclose(val, expected)

    def test_reduction(self):
        gt = GT
        pred = PRED
        eps = EPS

        dim = 1
        expected = 1 - (2 * (np.sum(gt * pred, axis=dim) + eps) / (np.sum(gt, axis=dim) + np.sum(pred, axis=dim) + eps))

        loss = DiceLoss(reduction=None, eps=eps)
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert val.shape == (3, 5), val.shape
        assert np.allclose(val, expected)

        loss = DiceLoss(reduction="mean", eps=eps)
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert np.isscalar(val), val
        assert np.allclose(val, np.mean(expected))

        loss = DiceLoss(reduction="sum", eps=eps)
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert np.isscalar(val), val
        assert np.allclose(val, np.sum(expected))
    
    def test_weight(self):
        gt = GT
        pred = PRED
        eps = EPS

        weights = np.asarray([1., 2., 3., 4., 5.])
        dim = 1
        dsc_loss = 1 - (2 * (np.sum(gt * pred, axis=dim) + eps) / (np.sum(gt, axis=dim) + np.sum(pred, axis=dim) + eps))
        dsc_loss[dsc_loss < eps] = 0
        expected = weights[np.newaxis, ...] * dsc_loss
        loss = DiceLoss(weights=weights, reduction=None, eps=eps)
        val = to_numpy(loss(K.constant(gt), K.constant(pred)))
        assert np.allclose(val[2], expected[2])
    
    def test_reproducible(self):
        """
        Select loss functions in `medsegpy.losses` can be represented
        with new DiceLoss class.
        """
        gt = K.constant(GT)
        pred = K.constant(PRED)
        weights = np.asarray([1., 2., 3., 4., 5.])

        # medsegpy.losses.dice_loss
        expected = to_numpy(dice_loss(gt, pred))
        loss = DiceLoss(reduction=None, flatten="channel")
        val = to_numpy(loss(gt, pred))
        assert np.allclose(val, expected)

        # medsegpy.losses.multi_class_dice_loss
        multi_class = multi_class_dice_loss(weights=weights, reduce="mean")
        expected = to_numpy(multi_class(gt, pred))
        loss = DiceLoss(weights=weights, reduction="mean", flatten="batch")
        val = to_numpy(loss(gt, pred))
        assert np.allclose(val, expected)

        multi_class = multi_class_dice_loss(reduce="class")
        expected = to_numpy(multi_class(gt, pred))
        loss = DiceLoss(reduction=None, flatten="batch")
        val = to_numpy(loss(gt, pred))
        assert np.allclose(val, expected)

        # medsegpy.losses.avg_dice_loss
        avg_dice = avg_dice_loss()
        expected = to_numpy(avg_dice(gt, pred))
        loss = DiceLoss(reduction="mean", flatten=False)
        val = to_numpy(loss(gt, pred))
        assert np.allclose(val, expected)

        avg_dice = avg_dice_loss(weights=weights)
        expected = to_numpy(avg_dice(gt, pred))
        loss = DiceLoss(weights=weights, reduction="mean", flatten=False)
        val = to_numpy(loss(gt, pred))
        assert np.allclose(val, expected)


if __name__ == "__main__":
    unittest.main()