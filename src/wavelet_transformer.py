# Wavelet transform and inverse functions

# src/wavelet_transformer.py
"""
WaveletTransformer Module
Applies wavelet decomposition and reconstruction for signal/data compression and expansion.
Supports integration with graphical representation and CorePatternLearner.
"""

import pywt
import numpy as np

class WaveletTransformer:
    def __init__(self, wavelet='db1', level=3):
        self.wavelet = wavelet
        self.level = level

    def decompose(self, signal):
        """
        Decomposes a 1D signal using wavelet transform.
        Returns list of coefficient arrays.
        """
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        return coeffs

    def reconstruct(self, coeffs):
        """
        Reconstructs a signal from wavelet coefficients.
        """
        return pywt.waverec(coeffs, self.wavelet)

    def compress_coeffs(self, coeffs, fibonacci_ratio=0.618):
        """
        Compress wavelet coefficients by zeroing out values below a Fibonacci-scaled threshold.
        """
        compressed_coeffs = []
        for i, c in enumerate(coeffs):
            threshold = np.percentile(np.abs(c), (1 - fibonacci_ratio) * 100)
            compressed_c = np.where(np.abs(c) >= threshold, c, 0)
            compressed_coeffs.append(compressed_c)
        return compressed_coeffs

    def expand_coeffs(self, compressed_coeffs):
        """
        Optionally reintroduce structure using an expansion ratio (currently identity).
        You can add noise or learned interpolation here later.
        """
        return compressed_coeffs

    def get_core_representation(self, signal):
        """
        Full core pattern extraction from raw signal:
        - Wavelet Decompose
        - Fibonacci Compress
        """
        coeffs = self.decompose(signal)
        compressed = self.compress_coeffs(coeffs)
        return compressed

    def restore_signal(self, compressed):
        """
        Full inverse process to restore signal from compressed coefficients.
        """
        return self.reconstruct(compressed)

# Example usage (for test purposes only):
if __name__ == '__main__':
    sample = np.sin(np.linspace(0, 2 * np.pi, 128))
    wt = WaveletTransformer(wavelet='db4', level=4)
    core = wt.get_core_representation(sample)
    restored = wt.restore_signal(core)
    print("Original vs Reconstructed (first 5):")
    print(sample[:5])
    print(restored[:5])
