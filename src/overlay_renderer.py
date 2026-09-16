"""
overlay_renderer.py
--------------------
Takes the solved 9x9 grid and draws the newly-filled-in digits back onto
the *original* input photo (not just the flat warped square), using the
inverse perspective matrix computed during grid detection. This is what
turns "a 9x9 array of numbers" back into a genuinely useful CV result:
a picture of the original puzzle with the solution overlaid in place.
"""

from __future__ import annotations
import cv2
import numpy as np

from .grid_detector import DetectedGrid, WARPED_SIZE, CELL_SIZE

SOLUTION_COLOR = (0, 200, 0)  # Green (BGR), to visually distinguish solved digits from clues
FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_solution(
    original_image: np.ndarray,
    detected: DetectedGrid,
    given_grid: list[list[int]],
    solved_grid: list[list[int]],
) -> np.ndarray:
    """
    Returns a copy of `original_image` with the solver's newly-filled
    digits drawn in place, respecting the original photo's perspective.
    """
    overlay_layer = np.zeros_like(detected.warped_color)

    for row in range(9):
        for col in range(9):
            if given_grid[row][col] != 0:
                continue  # Only draw digits the solver filled in, not the original clues

            digit = solved_grid[row][col]
            _draw_digit_in_cell(overlay_layer, row, col, digit)

    # Warp the overlay (and a matching mask) back into the original photo's perspective
    h, w = original_image.shape[:2]
    warped_back = cv2.warpPerspective(overlay_layer, detected.inverse_matrix, (w, h))

    mask = np.any(overlay_layer > 0, axis=2).astype(np.uint8) * 255
    mask_back = cv2.warpPerspective(mask, detected.inverse_matrix, (w, h))
    mask_back = cv2.threshold(mask_back, 10, 255, cv2.THRESH_BINARY)[1]

    result = original_image.copy()
    mask_bool = mask_back.astype(bool)
    result[mask_bool] = warped_back[mask_bool]

    return result


def _draw_digit_in_cell(canvas: np.ndarray, row: int, col: int, digit: int) -> None:
    text = str(digit)
    font_scale = CELL_SIZE / 45.0
    thickness = max(1, CELL_SIZE // 25)

    (text_w, text_h), _ = cv2.getTextSize(text, FONT, font_scale, thickness)
    center_x = col * CELL_SIZE + CELL_SIZE // 2
    center_y = row * CELL_SIZE + CELL_SIZE // 2
    origin = (center_x - text_w // 2, center_y + text_h // 2)

    cv2.putText(canvas, text, origin, FONT, font_scale, SOLUTION_COLOR, thickness, cv2.LINE_AA)


def save_flat_grid_visualization(detected: DetectedGrid, solved_grid: list[list[int]], given_grid: list[list[int]], path: str) -> None:
    """Also saves a clean, flat top-down image of the solved grid (useful for debugging/reports)."""
    canvas = detected.warped_color.copy()

    # Draw grid lines for readability
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        cv2.line(canvas, (i * CELL_SIZE, 0), (i * CELL_SIZE, WARPED_SIZE), (0, 0, 0), thickness)
        cv2.line(canvas, (0, i * CELL_SIZE), (WARPED_SIZE, i * CELL_SIZE), (0, 0, 0), thickness)

    for row in range(9):
        for col in range(9):
            if given_grid[row][col] == 0:
                _draw_digit_in_cell(canvas, row, col, solved_grid[row][col])

    cv2.imwrite(path, canvas)
