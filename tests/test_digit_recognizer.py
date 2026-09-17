"""Unit tests for src/digit_recognizer.py."""

import numpy as np
import cv2
import pytest

from src.digit_recognizer import DigitRecognizer, _isolate_digit
from src.exceptions import DigitRecognitionError


def _make_cell_with_digit(digit: str, size=50) -> np.ndarray:
    cell = np.full((size, size), 255, dtype=np.uint8)
    cv2.putText(cell, digit, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    return cell


def _make_blank_cell(size=50) -> np.ndarray:
    return np.full((size, size), 255, dtype=np.uint8)


def test_isolate_digit_returns_none_for_blank_cell():
    assert _isolate_digit(_make_blank_cell()) is None


def test_isolate_digit_returns_crop_for_filled_cell():
    crop = _isolate_digit(_make_cell_with_digit("7"))
    assert crop is not None
    assert crop.shape == (28, 28)


def test_recognizer_raises_on_missing_model_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.pkl"
    with pytest.raises(DigitRecognitionError):
        DigitRecognizer(model_path=missing_path)


def test_recognize_grid_rejects_wrong_cell_count():
    recognizer = DigitRecognizer()  # Uses the default trained model
    with pytest.raises(DigitRecognitionError):
        recognizer.recognize_grid([_make_blank_cell()] * 10)  # Not 81 cells


def test_recognize_grid_returns_9x9_structure():
    recognizer = DigitRecognizer()
    cells = [_make_blank_cell() for _ in range(81)]
    result = recognizer.recognize_grid(cells)

    assert len(result) == 9
    assert all(len(row) == 9 for row in result)
    assert all(v == 0 for row in result for v in row)  # All-blank input -> all zeros
