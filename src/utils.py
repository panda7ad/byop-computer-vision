"""
utils.py
--------
Shared helpers used across modules: logging setup, image I/O, and small
geometry utilities for working with the four corner points of the
detected Sudoku grid.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

from .exceptions import ImageLoadError


def get_logger(name: str, log_file: str = "sudoku_solver.log", verbose: bool = False) -> logging.Logger:
    """
    Returns a configured logger that writes to both the console and a
    rotating-free log file (kept simple on purpose for a CLI tool).
    Calling this repeatedly with the same `name` will not duplicate handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Non-fatal: if the log file can't be created (e.g. read-only dir),
        # fall back to console-only logging rather than crashing the CLI.
        logger.warning("Could not open log file '%s'; continuing with console logging only.", log_file)

    return logger


def load_image(path: str) -> np.ndarray:
    """Loads an image from disk as a BGR numpy array, raising a clear error if it fails."""
    file_path = Path(path)
    if not file_path.exists():
        raise ImageLoadError(f"Input image not found: '{path}'")

    image = cv2.imread(str(file_path))
    if image is None:
        raise ImageLoadError(
            f"Could not decode '{path}' as an image. "
            "Supported formats: .jpg, .jpeg, .png, .bmp."
        )
    return image


def order_corner_points(points: np.ndarray) -> np.ndarray:
    """
    Given 4 unordered (x, y) points forming a quadrilateral, returns them
    ordered as [top-left, top-right, bottom-right, bottom-left].

    Uses the standard sum/difference trick:
      - top-left has the smallest (x + y)
      - bottom-right has the largest (x + y)
      - top-right has the smallest (y - x)
      - bottom-left has the largest (y - x)
    """
    points = points.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).flatten()

    ordered[0] = points[np.argmin(sums)]  # top-left
    ordered[2] = points[np.argmax(sums)]  # bottom-right
    ordered[1] = points[np.argmin(diffs)]  # top-right
    ordered[3] = points[np.argmax(diffs)]  # bottom-left

    return ordered


def ensure_dir(path: str) -> None:
    """Creates a directory (including parents) if it doesn't already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
