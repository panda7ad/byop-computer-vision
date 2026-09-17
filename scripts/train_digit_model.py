"""
train_digit_model.py

Generates a synthetic dataset of printed Sudoku digits (1-9; Sudoku has
no 0/blank digit glyph to classify -- blanks are detected separately by
digit_recognizer.py via a pixel-density check) and trains an SVM
classifier on HOG features to recognize them.

Run:
    python scripts/train_digit_model.py"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.digit_features import extract_features, DIGIT_IMAGE_SIZE  # noqa: E402

RANDOM_SEED = 42
SAMPLES_PER_VARIANT = 30  # augmented copies generated per (digit, font) combo
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "digit_classifier.pkl"

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_COMPLEX_SMALL,
]
DIGITS = list(range(1, 10))  # Sudoku uses 1-9 only


def _render_digit(digit: int, font, rng: np.random.Generator) -> np.ndarray:
    """Renders a single digit onto a blank canvas with randomized augmentation."""
    canvas_size = 64
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    scale = rng.uniform(1.6, 2.4)
    thickness = int(rng.integers(2, 5))
    org = (int(rng.integers(10, 18)), int(rng.integers(44, 54)))

    cv2.putText(canvas, str(digit), org, font, scale, 255, thickness, cv2.LINE_AA)

    # Random rotation to simulate a slightly tilted photo
    angle = rng.uniform(-12, 12)
    center = (canvas_size // 2, canvas_size // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    canvas = cv2.warpAffine(canvas, rot_matrix, (canvas_size, canvas_size))

    # Mild Gaussian noise to simulate camera/scan artifacts
    noise = rng.normal(0, 8, canvas.shape).astype(np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Occasional slight blur, to mimic the adaptive-threshold + resize
    # softening that real cell crops go through in the actual pipeline.
    if rng.random() < 0.5:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
    _, canvas = cv2.threshold(canvas, 90, 255, cv2.THRESH_BINARY)

    # Crop to bounding box of the ink, then pad by a randomized amount
    # (matching the *range* of padding produced by digit_recognizer's
    # runtime crop, rather than one fixed value) before resizing.
    coords = cv2.findNonZero(canvas)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = max(2, int(rng.uniform(0.12, 0.24) * max(w, h)))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(canvas_size, x + w + pad), min(canvas_size, y + h + pad)
        canvas = canvas[y0:y1, x0:x1]

    return cv2.resize(canvas, (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE))


def build_dataset(rng: np.random.Generator):
    images, labels = [], []
    for digit in DIGITS:
        for font in FONTS:
            for _ in range(SAMPLES_PER_VARIANT):
                images.append(_render_digit(digit, font, rng))
                labels.append(digit)
    return images, labels


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    print(f"Generating synthetic dataset ({len(DIGITS)} digits x {len(FONTS)} fonts x "
          f"{SAMPLES_PER_VARIANT} augmented samples)...")
    images, labels = build_dataset(rng)
    print(f"Total samples: {len(images)}")

    features = np.array([extract_features(img) for img in images])
    labels = np.array(labels)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )

    print("Training SVM (RBF kernel) classifier on HOG features...")
    model = SVC(kernel="rbf", C=10, gamma="scale", random_state=RANDOM_SEED)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nHeld-out test accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, predictions, digits=3))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
