"""Transparent post-hoc baseline adapters for QRT comparisons."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.isotonic import IsotonicRegression

from models.predictions import GaussianMixturePrediction


@dataclass(frozen=True)
class QuantileRecalibrator:
    """Monotone PIT-to-PIT map fitted on a calibration split."""

    source: np.ndarray
    target: np.ndarray

    def map(self, pit: np.ndarray) -> np.ndarray:
        """Map PIT values through the fitted monotone interpolation."""
        return np.clip(PchipInterpolator(self.source, self.target, extrapolate=True)(pit), 0.0, 1.0)


def fit_quantile_recalibrator(prediction: GaussianMixturePrediction, targets: np.ndarray, levels: int = 100) -> QuantileRecalibrator:
    """Fit the standard empirical quantile-recalibration map (QR baseline)."""
    pit = prediction.cdf_1d(targets)
    source = np.linspace(0.0, 1.0, levels + 2)[1:-1]
    empirical = np.array([(pit <= level).mean() for level in source])
    isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True).fit(source, empirical)
    target = np.clip(isotonic.predict(source), 0.0, 1.0)
    return QuantileRecalibrator(np.r_[0.0, source, 1.0], np.r_[0.0, target, 1.0])


def apply_pit_map(prediction: GaussianMixturePrediction, targets: np.ndarray, calibrator: QuantileRecalibrator) -> np.ndarray:
    """Return recalibrated PIT values; full distribution inversion is data-dependent."""
    return calibrator.map(prediction.cdf_1d(targets))


def baseline_registry() -> dict[str, str]:
    """Name the requested baselines and their source provenance."""
    return {"QR": "empirical PIT quantile recalibration", "QRC": "upstream QRT post-hoc recalibration", "QRTC": "upstream QRT calibration training", "QREGC": "upstream QRT regularized calibration"}
