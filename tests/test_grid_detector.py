"""Unit tests for src/grid_detector.py and src/utils.py."""

import numpy as np
import cv2
import pytest

from src import grid_detector
from src.utils import order_corner_points, load_image
from src.exceptions import GridNotFoundError, ImageLoadError


def test_order_corner_points_sorts_correctly():
    # Deliberately shuffled/unordered input
    shuffled = np.array([[100, 100], [0, 0], [100, 0], [0, 100]], dtype=np.float32)
    ordered = order_corner_points(shuffled)

    top_left, top_right, bottom_right, bottom_left = ordered
    assert list(top_left) == [0, 0]
    assert list(top_right) == [100, 0]
    assert list(bottom_right) == [100, 100]
    assert list(bottom_left) == [0, 100]


def test_load_image_missing_file_raises():
    with pytest.raises(ImageLoadError):
        load_image("this_file_definitely_does_not_exist.jpg")


def _make_blank_grid_image(size=450):
    """Draws a plain, perfectly axis-aligned 9x9 grid for a controlled unit test."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cell = size // 9
    for i in range(10):
        thickness = 4 if i % 3 == 0 else 1
        cv2.line(img, (i * cell, 0), (i * cell, size), (0, 0, 0), thickness)
        cv2.line(img, (0, i * cell), (size, i * cell), (0, 0, 0), thickness)
    # Pad with a plain background so the grid isn't touching the image border
    padded = np.full((size + 100, size + 100, 3), 220, dtype=np.uint8)
    padded[50:50 + size, 50:50 + size] = img
    return padded


def test_detect_grid_on_clean_synthetic_image():
    image = _make_blank_grid_image()
    detected = grid_detector.detect_grid(image)

    assert detected.warped.shape == (grid_detector.WARPED_SIZE, grid_detector.WARPED_SIZE)
    assert detected.corners.shape == (4, 2)


def test_detect_grid_raises_on_image_with_no_grid():
    blank = np.full((300, 300, 3), 255, dtype=np.uint8)  # Featureless white image
    with pytest.raises(GridNotFoundError):
        grid_detector.detect_grid(blank)


def test_split_into_cells_returns_81_cells():
    image = _make_blank_grid_image()
    detected = grid_detector.detect_grid(image)
    cells = grid_detector.split_into_cells(detected.warped)

    assert len(cells) == 81
    expected_cell_size = grid_detector.WARPED_SIZE // 9
    for cell in cells:
        assert cell.shape == (expected_cell_size, expected_cell_size)
