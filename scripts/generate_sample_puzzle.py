"""
generate_sample_puzzle.py
--------------------------
Renders a synthetic "photographed" Sudoku puzzle image, used as the
demo/test input for the pipeline (sample_images/sample_sudoku.jpg).

This script exists so the whole project is reproducible end-to-end
without needing a real camera photo: grid detection, digit recognition,
solving, and overlay rendering can all be exercised and verified against
a known puzzle with a known solution.

Run:
    python scripts/generate_sample_puzzle.py
"""

from pathlib import Path
import cv2
import numpy as np

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "sample_images" / "sample_sudoku.jpg"

# A well-known easy puzzle (0 = blank cell)
PUZZLE = [
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

GRID_PX = 450
CELL_PX = GRID_PX // 9
CANVAS_PX = 640  # Larger canvas so the grid doesn't fill the whole frame (like a real photo)


def render_flat_grid() -> np.ndarray:
    grid_img = np.full((GRID_PX, GRID_PX), 255, dtype=np.uint8)

    for i in range(10):
        thickness = 4 if i % 3 == 0 else 1
        cv2.line(grid_img, (i * CELL_PX, 0), (i * CELL_PX, GRID_PX), 0, thickness)
        cv2.line(grid_img, (0, i * CELL_PX), (GRID_PX, i * CELL_PX), 0, thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(9):
        for col in range(9):
            digit = PUZZLE[row][col]
            if digit == 0:
                continue
            text = str(digit)
            (tw, th), _ = cv2.getTextSize(text, font, 1.3, 2)
            cx = col * CELL_PX + CELL_PX // 2 - tw // 2
            cy = row * CELL_PX + CELL_PX // 2 + th // 2
            cv2.putText(grid_img, text, (cx, cy), font, 1.3, 0, 2, cv2.LINE_AA)

    return cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)


def place_on_photo_like_background(grid_img: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)

    # Textured "table" background
    background = np.full((CANVAS_PX, CANVAS_PX, 3), 200, dtype=np.uint8)
    noise = rng.integers(-15, 15, background.shape, dtype=np.int16)
    background = np.clip(background.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Mild random perspective distortion of the grid, as if photographed at a slight angle
    src = np.float32([[0, 0], [GRID_PX, 0], [GRID_PX, GRID_PX], [0, GRID_PX]])
    jitter = 18
    dst = src + rng.uniform(-jitter, jitter, src.shape).astype(np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped_grid = cv2.warpPerspective(
        grid_img, matrix, (GRID_PX, GRID_PX), borderValue=(200, 200, 200)
    )

    offset = (CANVAS_PX - GRID_PX) // 2
    canvas = background.copy()
    canvas[offset:offset + GRID_PX, offset:offset + GRID_PX] = warped_grid

    canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    return canvas


def main():
    flat_grid = render_flat_grid()
    photo_like = place_on_photo_like_background(flat_grid)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), photo_like)
    print(f"Sample puzzle image saved to: {OUTPUT_PATH}")
    print("Ground-truth puzzle (0 = blank):")
    for row in PUZZLE:
        print(row)


if __name__ == "__main__":
    main()
