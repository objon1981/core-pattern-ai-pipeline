# src/fibonacci_compressor.py

import numpy as np

class FibonacciCompressor:
    """
    Compresses and decompresses vectors based on Fibonacci ratio scaling.
    """

    def __init__(self):
        self.fib_sequence = self._generate_fib_sequence(100)

    def _generate_fib_sequence(self, n):
        fib = [1, 1]
        for _ in range(2, n):
            fib.append(fib[-1] + fib[-2])
        return np.array(fib)

    def compress(self, data: np.ndarray, scale: int = 10) -> np.ndarray:
        """
        Compresses data by scaling with Fibonacci ratios.
        """
        fib = self.fib_sequence[:len(data)] / self.fib_sequence[scale]
        return data * fib

    def decompress(self, compressed_data: np.ndarray, scale: int = 10) -> np.ndarray:
        """
        Decompresses data by reversing Fibonacci ratio scaling.
        """
        fib = self.fib_sequence[:len(compressed_data)] / self.fib_sequence[scale]
        return compressed_data / fib
