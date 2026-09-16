"""
digit_recognizer.py
--------------------
Module 2: Given the 81 cell images produced by grid_detector.split_into_cells,
determines which cells are empty and classifies the digit in every
non-empty cell using the SVM model trained by scripts/train_digit_model.py.

Two-stage approach per cell:
  1. Emptiness check -- a cheap pixel-density test on the thresholded
     cell. Most of a Sudoku image is empty cells, so filtering these out
     early avoids running the classifier (and its false positives) on
     grid-line noise.
  2. Digit isolation + classification -- for non-empty cells, the
     digit's contour is isolated from the cell's border/grid-line noise,
     cropped, normalized, and passed through the HOG + SVM pipeline.
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
import joblib

from .exceptions import DigitRecognitionError
from .digit_features import extract_features, DIGIT_IMAGE_SIZE

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digit_classifier.pkl"

# A cell is considered empty if fewer than this fraction of its pixels
# (after thresholding and border-cropping) are "ink".
EMPTY_CELL_PIXEL_RATIO = 0.03
# Fraction of each cell's border to discard before analysis, to avoid
# picking up the grid lines themselves as "ink".
BORDER_CROP_RATIO = 0.10
# After thresholding, an extra thin frame (in pixels) is zeroed out to
# remove any grid-line fragments still hugging the interior's edge --
# a real digit stroke never touches the very edge of its cell.
EDGE_CLEAR_PX = 1


class DigitRecognizer:
    """Loads the trained classifier once and reuses it across cells/images."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        model_path = Path(model_path)
        if not model_path.exists():
            raise DigitRecognitionError(
                f"Digit classifier model not found at '{model_path}'. "
                "Run `python scripts/train_digit_model.py` first to train and save it."
            )
        try:
            self.model = joblib.load(model_path)
        except Exception as exc:  # noqa: BLE001 - surfaced as a domain-specific error
            raise DigitRecognitionError(f"Failed to load digit classifier: {exc}") from exc

    def recognize_grid(self, cells: list[np.ndarray]) -> list[list[int]]:
        """
        Takes the 81 cell images (row-major order, as produced by
        grid_detector.split_into_cells) and returns a 9x9 grid of ints,
        with 0 for cells recognized as empty.
        """
        if len(cells) != 81:
            raise DigitRecognitionError(f"Expected 81 cell images, got {len(cells)}.")

        flat_grid = [self._recognize_cell(cell) for cell in cells]
        return [flat_grid[row * 9:(row + 1) * 9] for row in range(9)]

    def _recognize_cell(self, cell: np.ndarray) -> int:
        digit_crop = _isolate_digit(cell)
        if digit_crop is None:
            return 0  # Empty cell

        features = extract_features(digit_crop).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        return int(prediction)


def _isolate_digit(cell: np.ndarray) -> np.ndarray | None:
    """
    Returns a normalized DIGIT_IMAGE_SIZE x DIGIT_IMAGE_SIZE crop of the
    digit in `cell`, or None if the cell is empty.
    """
    h, w = cell.shape[:2]
    bx, by = int(w * BORDER_CROP_RATIO), int(h * BORDER_CROP_RATIO)
    interior = cell[by:h - by, bx:w - bx]

    if interior.size == 0:
        return None

    _, thresh = cv2.threshold(interior, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Zero out a thin frame around the edge to strip leftover grid-line
    # fragments that survived the border crop (real digits are centered
    # and never touch the very edge of their cell).
    e = EDGE_CLEAR_PX
    thresh[:e, :] = 0
    thresh[-e:, :] = 0
    thresh[:, :e] = 0
    thresh[:, -e:] = 0

    ink_ratio = np.count_nonzero(thresh) / thresh.size
    if ink_ratio < EMPTY_CELL_PIXEL_RATIO:
        return None

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Grid-line remnants that survived the border crop tend to be thin
    # slivers touching the edge of the crop -- filter those out before
    # doing anything else, rather than relying solely on a large border
    # crop (which risks shaving off part of a digit stroke that sits
    # close to the cell's edge, e.g. the top bar of a "5").
    contours = [c for c in contours if not _looks_like_grid_line(c, thresh.shape)]
    if not contours:
        return None

    # A single digit's stroke can split into more than one contour (e.g. an
    # "8" or "6" whose loop closure got broken by thresholding), so instead
    # of keeping only the single largest contour, we take the union bounding
    # box of every contour large enough to plausibly be part of a digit --
    # this avoids silently chopping off part of the glyph.
    largest_area = max(cv2.contourArea(c) for c in contours)
    significant = [c for c in contours if cv2.contourArea(c) >= 0.15 * largest_area]

    x, y, x1, y1 = thresh.shape[1], thresh.shape[0], 0, 0
    for c in significant:
        cx, cy, cw, ch = cv2.boundingRect(c)
        x, y = min(x, cx), min(y, cy)
        x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
    cw, ch = x1 - x, y1 - y

    # Reject tiny regions -- almost always leftover grid-line fragments,
    # not an actual digit stroke.
    if cw * ch < 0.02 * thresh.size:
        return None

    # Pad around the tight bounding box before resizing, matching the
    # proportional padding used when generating the training data --
    # HOG features are position/scale sensitive, so keeping the same
    # amount of "breathing room" around the digit at both training and
    # inference time matters for the classifier to generalize.
    pad = max(2, int(0.18 * max(cw, ch)))
    y0, y1 = max(0, y - pad), min(thresh.shape[0], y + ch + pad)
    x0, x1 = max(0, x - pad), min(thresh.shape[1], x + cw + pad)

    digit = thresh[y0:y1, x0:x1]
    return cv2.resize(digit, (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE))


def _looks_like_grid_line(contour, shape) -> bool:
    """
    Heuristic for 'this contour is a leftover grid-line fragment, not part
    of a digit': it touches the crop's outer edge and is much wider/taller
    than it is thick.
    """
    x, y, w, h = cv2.boundingRect(contour)
    touches_edge = x <= 1 or y <= 1 or (x + w) >= shape[1] - 1 or (y + h) >= shape[0] - 1
    if not touches_edge:
        return False

    long_and_thin = (w > 0.5 * shape[1] and h < 0.2 * shape[0]) or (h > 0.5 * shape[0] and w < 0.2 * shape[1])
    return long_and_thin
