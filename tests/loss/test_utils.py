import unittest

import numpy as np
import keras.backend as K

from medsegpy.loss.utils import get_activation, get_shape, reduce_tensor, to_numpy

try:
    import tf.keras.activations as activations
except ImportError:
    import keras.activations as activations

X = np.asarray([
    [1,2,3,4,5],
    [6,7,8,9,10],
])
WEIGHTS = np.asarray([[1,2,3,4,5]])


class TestGetShape(unittest.TestCase):
    def test_get_shape(self):
        x = K.constant(np.random.rand(2,5,9))
        assert get_shape(x) == (2,5,9)


class TestReduceTensor(unittest.TestCase):
    def test_basic_reduction(self):
        x = K.constant(X)

        val = to_numpy(reduce_tensor(x, reduction="mean"))
        assert np.allclose(val, np.mean(X))

        val = to_numpy(reduce_tensor(x, reduction="sum"))
        assert np.allclose(val, np.sum(X))

        val = to_numpy(reduce_tensor(x, reduction="none"))
        assert np.allclose(val, X)

    def test_weight(self):
        """Broadcasting weights should not affect weighted reduction calculations."""
        x = K.constant(X)  # Shape: (2, 5)
        
        w = np.asarray([1,2,3,4,5])  # Shape: (5,)
        weights = K.constant(w)

        val = to_numpy(reduce_tensor(x, weights=weights, reduction="mean"))
        assert np.allclose(val, np.mean(np.sum(w * X, axis=-1) / np.sum(WEIGHTS)))
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="sum"))
        assert np.allclose(val, np.sum(w * X))
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="none"))
        assert np.allclose(val, w * X)

        w = np.asarray([[1,2,3,4,5]])  # Shape: (1,5)
        weights = K.constant(w)

        val = to_numpy(reduce_tensor(x, weights=weights, reduction="mean"))
        assert np.allclose(val, np.mean(np.sum(w * X, axis=-1) / np.sum(WEIGHTS)))
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="sum"))
        assert np.allclose(val, np.sum(w * X))
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="none"))
        assert np.allclose(val, w * X)

        w = np.asarray([[1],[2]])  # Shape: (2,1)
        weights = K.constant(w)
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="mean"))
        assert np.allclose(val, np.sum(w * X) / (np.sum(w) * X.shape[1]))
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="sum"))
        assert np.allclose(val, np.sum(w * X))
        val = to_numpy(reduce_tensor(x, weights=weights, reduction="none"))
        assert np.allclose(val, w * X)


class TestGetActivation(unittest.TestCase):
    def test_str(self):
        name, fn, args = get_activation("sigmoid")
        assert name == "sigmoid"
        assert fn == activations.sigmoid
        assert "axis" not in args

        name, fn, args = get_activation("softmax")
        assert name == "softmax"
        assert fn == activations.softmax
        assert "axis" in args
    
    def test_callable(self):
        name, fn, args = get_activation(activations.sigmoid)
        assert name == "sigmoid"
        assert fn == activations.sigmoid
        assert "axis" not in args

        name, fn, args = get_activation(activations.softmax)
        assert name == "softmax"
        assert fn == activations.softmax
        assert "axis" in args


if __name__ == "__main__":
    unittest.main()