"""Unit tests for the proposed covariance-preserving map."""
from __future__ import annotations

import numpy as np

from calibration.nrc_cal import fit_nrc_calibrator
from models.predictions import gaussian_prediction


def test_calibrator_preserves_gaussian_mean_and_positivity() -> None:
    rng = np.random.default_rng(7)
    mean = rng.normal(size=(30, 1)); variance = np.ones((30, 1))
    prediction = gaussian_prediction(mean, variance)
    fitted = fit_nrc_calibrator(prediction, mean + rng.normal(size=(30, 1)), np.linspace(0, 1, 30))
    transformed = fitted.transform(prediction, np.linspace(0, 1, 30))
    assert np.allclose(transformed.mean, prediction.mean)
    assert np.all(np.linalg.eigvalsh(transformed.covariances) > 0)

