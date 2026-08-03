"""Evaluation metrics used consistently across NRC-Cal experiments."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from models.predictions import GaussianMixturePrediction


def probability_calibration_error(prediction: GaussianMixturePrediction, targets: np.ndarray, levels: int = 100) -> float:
    """Compute QRT-style PCE: mean absolute empirical CDF error over levels."""
    if levels < 2:
        raise ValueError("levels must be at least two")
    pit = prediction.cdf_1d(targets)
    alpha = np.linspace(1 / levels, 1.0, levels)
    return float(np.mean(np.abs((pit[:, None] <= alpha).mean(axis=0) - alpha)))


def negative_log_likelihood(prediction: GaussianMixturePrediction, targets: np.ndarray) -> float:
    """Return average negative mixture log likelihood."""
    return float(-prediction.logpdf(targets).mean())


def rmse(prediction: GaussianMixturePrediction, targets: np.ndarray) -> float:
    """Return root mean squared error of predictive mean."""
    return float(np.sqrt(np.mean((prediction.mean - np.asarray(targets)) ** 2)))


def mae(prediction: GaussianMixturePrediction, targets: np.ndarray) -> float:
    """Return mean absolute error of predictive mean."""
    return float(np.mean(np.abs(prediction.mean - np.asarray(targets))))


def coverage_and_sharpness(prediction: GaussianMixturePrediction, targets: np.ndarray, level: float = 0.9) -> tuple[float, float]:
    """Return univariate central-interval coverage and mean interval width."""
    if prediction.means.shape[2] != 1 or not (0 < level < 1):
        raise ValueError("Coverage is currently defined for univariate targets and 0 < level < 1")
    y = np.asarray(targets).reshape(-1)
    z = norm.ppf((1.0 + level) / 2.0)
    variance = prediction.total_covariance[:, 0, 0]
    half_width = z * np.sqrt(variance)
    mean = prediction.mean[:, 0]
    return float(np.mean(np.abs(y - mean) <= half_width)), float(np.mean(2.0 * half_width))


def crps_gaussian_moment_match(prediction: GaussianMixturePrediction, targets: np.ndarray) -> float:
    """Compute a documented moment-matched Gaussian CRPS for univariate mixtures."""
    if prediction.means.shape[2] != 1:
        raise ValueError("CRPS is implemented for univariate predictions")
    mu = prediction.mean[:, 0]
    sigma = np.sqrt(prediction.total_covariance[:, 0, 0])
    z = (np.asarray(targets).reshape(-1) - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(crps.mean())


def evaluate(prediction: GaussianMixturePrediction, targets: np.ndarray) -> dict[str, float]:
    """Compute all applicable core metrics in one stable result dictionary."""
    values = {"nll": negative_log_likelihood(prediction, targets), "rmse": rmse(prediction, targets), "mae": mae(prediction, targets)}
    if prediction.means.shape[2] == 1:
        coverage, sharpness = coverage_and_sharpness(prediction, targets)
        values.update({"pce": probability_calibration_error(prediction, targets), "crps": crps_gaussian_moment_match(prediction, targets), "coverage_90": coverage, "sharpness_90": sharpness})
    return values
