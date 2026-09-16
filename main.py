#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point for the Sudoku Vision Solver.

Usage
-----
    python main.py solve --image sample_images/sample_sudoku.jpg --output output/solved.jpg
    python main.py solve --image path/to/photo.jpg --debug
    python main.py train

Run `python main.py --help` or `python main.py solve --help` for all options.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import cv2

from src import grid_detector, sudoku_solver
from src.digit_recognizer import DigitRecognizer
from src.overlay_renderer import render_solution, save_flat_grid_visualization
from src.utils import get_logger, load_image, ensure_dir
from src.exceptions import SudokuVisionError

logger = get_logger("sudoku_vision_solver")


def run_solve(args: argparse.Namespace) -> int:
    start_time = time.perf_counter()
    ensure_dir(str(Path(args.output).parent))

    try:
        logger.info("Loading image: %s", args.image)
        image = load_image(args.image)

        logger.info("Detecting Sudoku grid...")
        detected = grid_detector.detect_grid(image)

        if args.debug:
            debug_path = str(Path(args.output).with_name("debug_warped_grid.jpg"))
            cv2.imwrite(debug_path, detected.warped_color)
            logger.debug("Saved warped grid for inspection: %s", debug_path)

        logger.info("Splitting grid into 81 cells and recognizing digits...")
        cells = grid_detector.split_into_cells(detected.warped)
        recognizer = DigitRecognizer(model_path=args.model)
        given_grid = recognizer.recognize_grid(cells)

        logger.info("Recognized puzzle:")
        for row in given_grid:
            logger.info(" ".join(str(v) if v else "." for v in row))

        logger.info("Solving puzzle...")
        solved_grid = sudoku_solver.solve(given_grid)

        logger.info("Rendering solution back onto the original image...")
        result_image = render_solution(image, detected, given_grid, solved_grid)
        cv2.imwrite(args.output, result_image)

        if args.debug:
            flat_path = str(Path(args.output).with_name("debug_solved_flat.jpg"))
            save_flat_grid_visualization(detected, solved_grid, given_grid, flat_path)
            logger.debug("Saved flat solved-grid visualization: %s", flat_path)

        elapsed = time.perf_counter() - start_time
        logger.info("Done in %.2f seconds. Result saved to: %s", elapsed, args.output)
        return 0

    except SudokuVisionError as exc:
        # Known, expected failure modes get a clean message instead of a traceback.
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - last-resort safety net for the CLI
        logger.error("Unexpected error: %s", exc)
        if args.verbose:
            raise
        return 1


def run_train(args: argparse.Namespace) -> int:
    # Deferred import: keeps `scripts/` out of the load path unless actually needed.
    sys.path.append(str(Path(__file__).resolve().parent))
    from scripts.train_digit_model import main as train_main
    train_main()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sudoku-vision-solver",
        description="Detects a Sudoku puzzle in a photo, solves it, and overlays the solution.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve_parser = subparsers.add_parser("solve", help="Solve a Sudoku puzzle from an image.")
    solve_parser.add_argument("--image", required=True, help="Path to the input puzzle image.")
    solve_parser.add_argument(
        "--output", default="output/solved.jpg", help="Path to save the solved image (default: output/solved.jpg)."
    )
    solve_parser.add_argument(
        "--model", default=None,
        help="Path to a trained digit classifier .pkl file (default: models/digit_classifier.pkl).",
    )
    solve_parser.add_argument(
        "--debug", action="store_true", help="Also save the warped grid and flat solved-grid images."
    )
    solve_parser.add_argument(
        "--verbose", action="store_true", help="Print full tracebacks on unexpected errors."
    )
    solve_parser.set_defaults(func=run_solve)

    train_parser = subparsers.add_parser(
        "train", help="(Re)train the digit classifier on a freshly generated synthetic dataset."
    )
    train_parser.set_defaults(func=run_train)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "command", None) == "solve" and args.model is None:
        from src.digit_recognizer import DEFAULT_MODEL_PATH
        args.model = str(DEFAULT_MODEL_PATH)

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
