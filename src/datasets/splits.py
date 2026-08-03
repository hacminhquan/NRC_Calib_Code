"""Split discovery and deterministic fallback splitting for tabular arrays."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class TabularSplits:
    """Non-overlapping train, validation, calibration, and test arrays."""

    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    calibration_x: np.ndarray
    calibration_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def split_tabular(x: np.ndarray, y: np.ndarray, seed: int = 0, ratios: tuple[float, float, float, float] = (0.65, 0.10, 0.15, 0.10)) -> TabularSplits:
    """Create reproducible train/validation/calibration/test splits without leakage."""
    if x.shape[0] != y.shape[0] or x.shape[0] < 8:
        raise ValueError("x and y must have the same length of at least eight")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("Split ratios must sum to one")
    indices = np.random.default_rng(seed).permutation(x.shape[0])
    ends = np.cumsum(np.asarray(ratios[:-1]) * x.shape[0]).astype(int)
    groups = np.split(indices, ends)
    if any(len(group) == 0 for group in groups):
        raise ValueError("Dataset is too small for requested split ratios")
    arrays: list[np.ndarray] = []
    for group in groups:
        arrays.extend([x[group], y[group]])
    return TabularSplits(*arrays)


def discover_splits(arrays: Mapping[str, np.ndarray]) -> TabularSplits:
    """Validate conventional `train_x` through `test_y` keys from an NPZ cache."""
    required = ("train_x", "train_y", "validation_x", "validation_y", "calibration_x", "calibration_y", "test_x", "test_y")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"Missing split arrays: {missing}")
    return TabularSplits(*(np.asarray(arrays[key]) for key in required))
