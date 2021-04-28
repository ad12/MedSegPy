import unittest

import keras.backend as K
import numpy as np
import scipy.special as sps

from medsegpy.losses import NaiveAdaRobLossComputer, multi_class_dice_loss
from medsegpy.utils import env


class TestNaiveAdaRobLossComputer(unittest.TestCase):
    def test_numpy(self):
        """Test to see if numpy works"""
        if not env.is_tf2():
            return
        loss = NaiveAdaRobLossComputer(
            multi_class_dice_loss(reduce="class", use_numpy=True), n_groups=4, robust_step_size=0.01
        )
        loss.training = True

        gt = np.random.rand(1, 5, 4) > 0
        pred = np.random.rand(1, 5, 4)

        _ = loss(gt, pred)

    def test_over_time(self):
        """Test that the weightage of the classes change over time.

        We simulate a case where, of the 3 classes (A, B, C), classes A and B will
        always have high accuracy (1) and therefore a loss of (0). However, class C will be random.
        We should see over time that class C gets the highest weight.
        """
        if not env.is_tf2():
            return
        num_classes = 3
        loss = NaiveAdaRobLossComputer(
            multi_class_dice_loss(reduce="class", use_numpy=True),
            n_groups=num_classes,
            robust_step_size=0.01,
        )
        loss.training = True

        for _i in range(1000):
            gt = np.random.rand(1, 5, num_classes) > 0
            pred = np.concatenate([gt[..., :2], np.random.rand(1, 5, 1)], axis=-1)
            _ = loss(gt, pred)

        weights = sps.softmax(loss.adv_probs_logits, axis=-1)
        assert all(weights[-1] > weights[i] for i in range(num_classes - 1))

    def test_keras(self):
        """Simple test to see if works with Keras tensors"""
        if not env.is_tf2():
            return
        num_classes = 3
        loss = NaiveAdaRobLossComputer(
            multi_class_dice_loss(reduce="class"), n_groups=num_classes, robust_step_size=0.01
        )
        loss.training = True

        for _i in range(10):
            gt = np.random.rand(1, 5, num_classes) > 0
            pred = np.concatenate([gt[..., :2], np.random.rand(1, 5, 1)], axis=-1)
            _ = loss(K.constant(gt), K.constant(pred))
