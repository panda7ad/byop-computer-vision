"""
grid_detector.py
-----------------
Module 1: Locates the outer boundary of a 9x9 Sudoku grid in a photo and
warps it into a flat, top-down square image, then slices that square
into 81 individual cell images.

Pipeline
--------
1. Grayscale + Gaussian blur (noise reduction)
2. Adaptive threshold (robust to uneven lighting, unlike a global threshold)
3. Find the largest 4-sided contour -> assumed to be the grid boundary
4. Order its corners and apply a perspective transform (cv2.warpPerspective)
5. Slice the resulting square into a 9x9 array of cell images
"""

from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .exceptions import GridNotFoundError
from .utils import order_corner_points

WARPED_SIZE = 450  # Output grid is WARPED_SIZE x WARPED_SIZE pixels
CELL_SIZE = WARPED_SIZE // 9
MIN_GRID_AREA_RATIO = 0.15  # Grid contour must cover at least this fraction of the image


@dataclass
class DetectedGrid:
    """Holds the results of grid detection for downstream modules."""
    warped: np.ndarray          # WARPED_SIZE x WARPED_SIZE grayscale grid image
    warped_color: np.ndarray    # Same, but color (for the final overlay)
    corners: np.ndarray         # The 4 ordered corner points found in the original image
    perspective_matrix: np.ndarray   # Forward transform (original -> warped)
    inverse_matrix: np.ndarray       # Inverse transform (warped -> original), for overlay


def _preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=11, C=2,
    )
    # Dilate slightly to close small gaps in the grid lines
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(thresh, kernel, iterations=1)


def _find_grid_contour(thresh: np.ndarray, image_area: float) -> np.ndarray:
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise GridNotFoundError("No contours found in the image at all.")

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < MIN_GRID_AREA_RATIO * image_area:
        raise GridNotFoundError(
            "Largest detected contour is too small to be a Sudoku grid. "
            "Try a clearer, more zoomed-in photo of the puzzle."
        )

    perimeter = cv2.arcLength(largest, True)

    # Slight irregularities in the contour (blur, compression, a not-quite-straight
    # edge) mean a single fixed epsilon doesn't reliably collapse the boundary to
    # exactly 4 points, so we search increasing approximation strengths until it does.
    approx = None
    for eps_fraction in np.arange(0.01, 0.06, 0.005):
        candidate = cv2.approxPolyDP(largest, eps_fraction * perimeter, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is None:
        # Fall back to the minimum-area bounding rectangle of the contour --
        # less precise than a true corner-detected quadrilateral, but robust.
        rect = cv2.minAreaRect(largest)
        approx = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.int32)

    return approx


def detect_grid(image: np.ndarray) -> DetectedGrid:
    """Detects the Sudoku grid in `image` and returns a warped top-down view."""
    height, width = image.shape[:2]
    thresh = _preprocess(image)
    contour = _find_grid_contour(thresh, image_area=height * width)
    corners = order_corner_points(contour)

    destination = np.array(
        [[0, 0], [WARPED_SIZE - 1, 0], [WARPED_SIZE - 1, WARPED_SIZE - 1], [0, WARPED_SIZE - 1]],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(corners, destination)
    inverse_matrix = cv2.getPerspectiveTransform(destination, corners)

    warped_color = cv2.warpPerspective(image, matrix, (WARPED_SIZE, WARPED_SIZE))
    warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)

    return DetectedGrid(
        warped=warped_gray,
        warped_color=warped_color,
        corners=corners,
        perspective_matrix=matrix,
        inverse_matrix=inverse_matrix,
    )


def split_into_cells(warped_gray: np.ndarray) -> list:
    """
    Splits the WARPED_SIZE x WARPED_SIZE grid image into a 9x9 (row-major)
    list of individual cell images.
    """
    cells = []
    for row in range(9):
        for col in range(9):
            y0, y1 = row * CELL_SIZE, (row + 1) * CELL_SIZE
            x0, x1 = col * CELL_SIZE, (col + 1) * CELL_SIZE
            cells.append(warped_gray[y0:y1, x0:x1])
    return cells
