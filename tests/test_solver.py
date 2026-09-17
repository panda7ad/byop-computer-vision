"""Unit tests for src/sudoku_solver.py -- pure logic, no image dependencies."""

import pytest

from src.sudoku_solver import (
    solve, is_valid_placement, is_valid_puzzle, is_valid_grid_shape, count_clues,
)
from src.exceptions import InvalidGridError, UnsolvableSudokuError

EASY_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

EASY_SOLUTION = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]


def test_solves_known_easy_puzzle():
    assert solve(EASY_PUZZLE) == EASY_SOLUTION


def test_does_not_mutate_input_by_default():
    original = [row[:] for row in EASY_PUZZLE]
    solve(EASY_PUZZLE)
    assert EASY_PUZZLE == original


def test_in_place_solve_mutates_input():
    puzzle_copy = [row[:] for row in EASY_PUZZLE]
    solve(puzzle_copy, in_place=True)
    assert puzzle_copy == EASY_SOLUTION


def test_already_solved_grid_returns_unchanged():
    assert solve(EASY_SOLUTION) == EASY_SOLUTION


def test_blank_grid_is_solvable():
    blank = [[0] * 9 for _ in range(9)]
    result = solve(blank)
    assert is_valid_puzzle(result)
    assert count_clues(result) == 81


def test_invalid_shape_raises():
    with pytest.raises(InvalidGridError):
        solve([[1, 2, 3]])  # Not 9x9


def test_conflicting_clues_raise_invalid_grid_error():
    bad_puzzle = [row[:] for row in EASY_PUZZLE]
    bad_puzzle[0][2] = 5  # Duplicate 5 in row 0 (already has a 5 at col 0)
    with pytest.raises(InvalidGridError):
        solve(bad_puzzle)


def test_unsolvable_puzzle_raises():
    # Box (rows 0-2, cols 0-2) is filled with 1-8, leaving only (2,2) empty --
    # the box's only remaining valid digit is 9. But row 2 already has a 9
    # elsewhere (col 5), so (2,2) has zero legal candidates: unsolvable.
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0], grid[0][1], grid[0][2] = 1, 2, 3
    grid[1][0], grid[1][1], grid[1][2] = 4, 5, 6
    grid[2][0], grid[2][1] = 7, 8
    grid[2][5] = 9

    with pytest.raises(UnsolvableSudokuError):
        solve(grid)


def test_is_valid_placement_detects_row_conflict():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 4
    assert not is_valid_placement(grid, 0, 5, 4)


def test_is_valid_placement_detects_column_conflict():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 4
    assert not is_valid_placement(grid, 5, 0, 4)


def test_is_valid_placement_detects_box_conflict():
    grid = [[0] * 9 for _ in range(9)]
    grid[0][0] = 4
    assert not is_valid_placement(grid, 2, 2, 4)


def test_is_valid_placement_allows_legal_move():
    grid = [[0] * 9 for _ in range(9)]
    assert is_valid_placement(grid, 4, 4, 7)


def test_is_valid_grid_shape_rejects_out_of_range_values():
    bad = [[0] * 9 for _ in range(9)]
    bad[0][0] = 10
    assert not is_valid_grid_shape(bad)


def test_count_clues():
    assert count_clues(EASY_PUZZLE) == sum(1 for row in EASY_PUZZLE for v in row if v != 0)
