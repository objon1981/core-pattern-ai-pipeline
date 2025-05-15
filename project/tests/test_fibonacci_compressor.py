# tests/test_fibonacci_compressor.py

import unittest
import numpy as np
from src.fibonacci_compressor import FibonacciCompressor

class TestFibonacciCompressor(unittest.TestCase):

    def setUp(self):
        self.compressor = FibonacciCompressor()
        self.test_array = np.array([10, 20, 30, 40, 50])

    def test_compression_decompression(self):
        compressed = self.compressor.compress(self.test_array, scale=10)
        decompressed = self.compressor.decompress(compressed, scale=10)
        np.testing.assert_almost_equal(decompressed, self.test_array, decimal=4)

    def test_length_consistency(self):
        compressed = self.compressor.compress(self.test_array, scale=10)
        self.assertEqual(len(compressed), len(self.test_array))

if __name__ == '__main__':
    unittest.main()