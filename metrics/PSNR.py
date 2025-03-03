# calculate peak signal to noise ration
# based on https://www.geeksforgeeks.org/python-peak-signal-to-noise-ratio-psnr/
from math import log10, sqrt
import numpy as np
from numpy.typing import NDArray


def PSNR(original: NDArray[np.uint8], upscaled: NDArray[np.uint8]) -> float:
    mse = np.mean((original - upscaled) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
