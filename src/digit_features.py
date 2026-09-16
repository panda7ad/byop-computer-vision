"""
digit_features.py
------------------
Shared feature extraction for the digit classifier, used identically at
both training time (scripts/train_digit_model.py) and inference time
(src/digit_recognizer.py) so the two never drift out of sync.

We use a Histogram of Oriented Gradients (HOG) descriptor rather than
raw pixel intensities: HOG captures the *shape* of the digit's strokes
and is far more tolerant of small shifts, thickness, and lighting
variation than comparing pixels directly.
"""

import cv2
import numpy as np

DIGIT_IMAGE_SIZE = 28  # All digit crops are normalized to 28x28 before feature extraction

_HOG = cv2.HOGDescriptor(
    _winSize=(28, 28),
    _blockSize=(14, 14),
    _blockStride=(7, 7),
    _cellSize=(7, 7),
    _nbins=9,
)


def extract_features(digit_image: np.ndarray) -> np.ndarray:
    """
    Computes a feature vector for a single digit image: a HOG descriptor
    (captures overall stroke shape) concatenated with four quadrant ink-
    density ratios (captures *where* the ink is concentrated). The
    quadrant densities specifically help separate shape pairs that HOG
    alone tends to confuse at this resolution, like 6 vs 9 (top-heavy
    loop vs bottom-heavy loop) and 3 vs 5 (which half is flat vs curved).
    Accepts any size and resizes internally, so callers don't need to
    worry about exact dimensions.
    """
    if digit_image.shape[:2] != (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE):
        digit_image = cv2.resize(digit_image, (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE))

    if digit_image.dtype != np.uint8:
        digit_image = digit_image.astype(np.uint8)

    hog_features = _HOG.compute(digit_image).flatten()
    quadrant_features = _quadrant_density_ratios(digit_image)
    return np.concatenate([hog_features, quadrant_features])


def _quadrant_density_ratios(digit_image: np.ndarray) -> np.ndarray:
    """Returns [top_ratio, bottom_ratio, left_ratio, right_ratio] of ink density."""
    mid = digit_image.shape[0] // 2
    total = np.count_nonzero(digit_image) + 1e-6  # avoid div-by-zero on a blank crop

    top = np.count_nonzero(digit_image[:mid, :]) / total
    bottom = np.count_nonzero(digit_image[mid:, :]) / total
    left = np.count_nonzero(digit_image[:, :mid]) / total
    right = np.count_nonzero(digit_image[:, mid:]) / total

    # Scaled up so this handful of features isn't drowned out by the
    # much longer HOG vector during distance/margin-based classification.
    return np.array([top, bottom, left, right]) * 10.0
