# Problem Statement

## Problem

Solving a Sudoku puzzle by hand is time-consuming, and manually
re-typing a printed puzzle into a solver app is tedious and error-prone.
Most existing "Sudoku scanner" apps are closed-source mobile apps that
give no visibility into *how* the recognition or solving actually
works — which makes them a poor fit for anyone who wants to understand
or extend the underlying computer vision pipeline.

**Sudoku Vision Solver** takes a single photo of a Sudoku puzzle and
automatically detects the grid, reads the printed digits, solves the
puzzle, and returns an image with the solution overlaid directly onto
the original photo — entirely from the command line, with every stage
of the pipeline implemented and documented as a separate, inspectable
module.

## Scope

**In scope:**
- Detecting a 9x9 Sudoku grid in a reasonably well-lit, mostly-frontal
  photo (scanned page, printed puzzle, screenshot, or a phone photo
  taken close to top-down)
- Recognizing printed (not handwritten) digits 1-9 in the grid
- Solving the puzzle with a backtracking algorithm
- Rendering the solution back onto the original image
- A from-scratch-trained digit classifier (SVM on HOG features), with
  its own synthetic data generation and training pipeline included, so
  the whole system is reproducible without external datasets
- A CLI interface with clear logging, error handling, and a debug mode

**Out of scope:**
- Handwritten digit recognition
- Extremely skewed, low-light, or partially-occluded photos
- A GUI or mobile app front-end
- Real-time video/webcam input (the pipeline operates on a single
  static image)

## Target Users

- Students or hobbyists who want to check their answer to a puzzle
  from a newspaper, book, or app without re-typing all 81 cells
- Anyone studying classical computer vision (contour detection,
  perspective transforms, HOG features) who wants a complete,
  real, end-to-end reference project rather than an isolated snippet
- Developers who want a CLI building block for a larger Sudoku-related
  tool (e.g. a batch puzzle-checker)

## High-Level Features

1. **Grid detection & perspective correction** — locates the puzzle's
   outer boundary in the photo and warps it into a flat, top-down
   square image, robust to a moderately tilted/rotated source photo.
2. **Digit recognition** — splits the flattened grid into 81 cells,
   determines which are empty, and classifies the digit in every
   filled cell using a trained SVM + HOG pipeline.
3. **Solving & overlay rendering** — solves the recognized puzzle with
   a backtracking search and draws the newly-solved digits back onto
   the original photo in its original perspective, visually
   distinguishing solved digits from the puzzle's original clues.
