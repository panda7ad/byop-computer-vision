"""
exceptions.py
-------------
Custom exception hierarchy for the Sudoku Vision Solver.

Using specific exception types (instead of letting raw OpenCV/numpy
errors bubble up) is part of this project's error-handling strategy:
each stage of the pipeline fails with a message that tells the CLI
user exactly what went wrong and, where possible, how to fix it.
"""


class SudokuVisionError(Exception):
    """Base class for all custom errors raised by this project."""


class ImageLoadError(SudokuVisionError):
    """Raised when the input image cannot be read from disk."""


class GridNotFoundError(SudokuVisionError):
    """Raised when a 9x9 Sudoku grid boundary cannot be located in the image."""


class DigitRecognitionError(SudokuVisionError):
    """Raised when the digit classifier model is missing or fails to load."""


class InvalidGridError(SudokuVisionError):
    """Raised when a recognized/parsed grid does not form a valid Sudoku puzzle."""


class UnsolvableSudokuError(SudokuVisionError):
    """Raised when the backtracking solver proves no solution exists."""
