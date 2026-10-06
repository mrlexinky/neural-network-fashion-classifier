"""Fashion-MNIST CSV loading utilities."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_fashion_csv(path: str | Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load label-first Fashion-MNIST CSV rows and normalise their pixels."""
    labels: list[int] = []
    images: list[list[int]] = []
    with Path(path).open(newline="") as source:
        reader = csv.reader(source)
        next(reader)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            labels.append(int(row[0]))
            images.append([int(pixel) for pixel in row[1:]])
    return np.asarray(images, dtype=np.float32) / 255.0, np.asarray(labels)
