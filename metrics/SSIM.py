# returns structural similarity
# based on https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity

import numpy as np
from numpy.typing import NDArray
from skimage.metrics import structural_similarity


def SSIM(original: NDArray[np.uint8], upscaled: NDArray[np.uint8]) -> float:
    score = structural_similarity(original, upscaled, channel_axis=2)
    return score
