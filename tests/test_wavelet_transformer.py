import unittest
import numpy as np
from src.wavelet_transformer import WaveletTransformer


class TestWaveletTransformer(unittest.TestCase):
    def setUp(self):
        self.signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        self.wt = WaveletTransformer(wavelet="db1", level=2)

    def test_forward_transform(self):
        coeffs = self.wt.forward(self.signal)
        self.assertIsInstance(coeffs, list)
        self.assertTrue(all(isinstance(c, np.ndarray) for c in coeffs))

    def test_inverse_transform(self):
        coeffs = self.wt.forward(self.signal)
        reconstructed = self.wt.inverse(coeffs)
        np.testing.assert_array_almost_equal(reconstructed, self.signal)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.wt.forward("not a signal")


if __name__ == '__main__':
    unittest.main()