"""
sudoku_solver.py
-----------------
Module 3: Constraint-checking + backtracking Sudoku solver.

This module is pure logic and has no dependency on OpenCV or image
processing, which keeps it independently unit-testable.

Grid representation
--------------------
A Sudoku grid is a 9x9 list of lists (or numpy array) of ints, where
0 represents an empty cell and 1-9 represent a filled cell.
"""

from __future__ import annotations
from typing import List, Optional
import copy

from .exceptions import InvalidGridError, UnsolvableSudokuError

Grid = List[List[int]]

BOX_SIZE = 3
GRID_SIZE = 9


def is_valid_grid_shape(grid: Grid) -> bool:
    """Checks that the grid is 9x9 and every cell is an int in [0, 9]."""
    if len(grid) != GRID_SIZE:
        return False
    for row in grid:
        if len(row) != GRID_SIZE:
            return False
        for val in row:
            if not isinstance(val, (int,)) or val < 0 or val > 9:
                return False
    return True


def is_valid_placement(grid: Grid, row: int, col: int, num: int) -> bool:
    """
    Returns True if placing `num` at (row, col) does not violate the
    row, column, or 3x3 box constraint.
    """
    # Row / column check
    for i in range(GRID_SIZE):
        if grid[row][i] == num and i != col:
            return False
        if grid[i][col] == num and i != row:
            return False

    # 3x3 box check
    box_row, box_col = BOX_SIZE * (row // BOX_SIZE), BOX_SIZE * (col // BOX_SIZE)
    for r in range(box_row, box_row + BOX_SIZE):
        for c in range(box_col, box_col + BOX_SIZE):
            if grid[r][c] == num and (r, c) != (row, col):
                return False

    return True


def is_valid_puzzle(grid: Grid) -> bool:
    """
    Validates that the *given* clues (non-zero cells) in the puzzle do not
    already violate Sudoku constraints. Does not check solvability.
    """
    if not is_valid_grid_shape(grid):
        raise InvalidGridError("Grid must be a 9x9 matrix of integers between 0 and 9.")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            val = grid[r][c]
            if val != 0 and not is_valid_placement(grid, r, c, val):
                return False
    return True


def _find_empty_cell(grid: Grid) -> Optional[tuple]:
    """
    Finds the next empty cell using the Minimum Remaining Values (MRV)
    heuristic: picks the empty cell with the fewest legal candidates,
    which prunes the backtracking search tree significantly faster than
    scanning left-to-right / top-to-bottom.
    """
    best_cell = None
    best_candidates = None

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r][c] == 0:
                candidates = [n for n in range(1, 10) if is_valid_placement(grid, r, c, n)]
                if best_candidates is None or len(candidates) < len(best_candidates):
                    best_cell, best_candidates = (r, c), candidates
                    if len(candidates) == 0:
                        # Dead end found early -- no point searching further
                        return best_cell, best_candidates
    return (best_cell, best_candidates) if best_cell else None


def solve(grid: Grid, in_place: bool = False) -> Grid:
    """
    Solves a Sudoku grid using backtracking search with an MRV heuristic.

    Parameters
    ----------
    grid : 9x9 list of lists of ints (0 = empty)
    in_place : if False (default), solves a deep copy and leaves the
        original grid untouched.

    Returns
    -------
    The solved 9x9 grid.

    Raises
    ------
    InvalidGridError: if the grid shape/values are invalid, or the given
        clues already break Sudoku's constraints.
    UnsolvableSudokuError: if no solution exists for the given clues.
    """
    if not is_valid_puzzle(grid):
        raise InvalidGridError("Puzzle's given clues violate Sudoku rules.")

    working_grid = grid if in_place else copy.deepcopy(grid)

    if not _backtrack(working_grid):
        raise UnsolvableSudokuError("No solution exists for the given puzzle.")

    return working_grid


def _backtrack(grid: Grid) -> bool:
    empty = _find_empty_cell(grid)
    if empty is None:
        return True  # No empty cells left -> solved

    (row, col), candidates = empty
    if not candidates:
        return False  # Dead end

    for num in candidates:
        grid[row][col] = num
        if _backtrack(grid):
            return True
        grid[row][col] = 0  # Undo (backtrack)

    return False


def count_clues(grid: Grid) -> int:
    """Returns the number of pre-filled (non-zero) cells in the grid."""
    return sum(1 for row in grid for val in row if val != 0)
