"""NRC-Cal's explicitly proposed closed-form frozen-model scale correction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import digamma

from models.predictions import GaussianMixturePrediction


@dataclass(frozen=True)
class NRCCalibrator:
    """Fitted coefficients for the proposed log-Mahalanobis scale map."""

    intercept: float
    geometry_slope: float
    distance_mean: float
    distance_std: float
    target_dimension: int
    ridge: float
    scale_min: float
    scale_max: float

    def scales(self, distances: np.ndarray) -> np.ndarray:
        """Compute bounded positive NRC-Cal scale factors for new examples."""
        values = np.asarray(distances, dtype=np.float64).reshape(-1)
        z = (values - self.distance_mean) / max(self.distance_std, np.finfo(float).eps)
        tau = float(digamma(self.target_dimension / 2.0) + np.log(2.0) - np.log(self.target_dimension))
        raw = np.exp(0.5 * (self.intercept + self.geometry_slope * z - tau))
        return np.clip(raw, self.scale_min, self.scale_max)

    def transform(self, prediction: GaussianMixturePrediction, distances: np.ndarray) -> GaussianMixturePrediction:
        """Apply the valid covariance-preserving Gaussian-mixture correction."""
        scale = self.scales(distances)
        if scale.shape[0] != prediction.means.shape[0]:
            raise ValueError("One NRC distance is required per prediction")
        center = prediction.mean[:, None, :]
        means = center + scale[:, None, None] * (prediction.means - center)
        covariances = scale[:, None, None, None] ** 2 * prediction.covariances
        return GaussianMixturePrediction(prediction.weights.copy(), means, covariances)


def mahalanobis_scale_residuals(prediction: GaussianMixturePrediction, targets: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    """Return proposed `q_i`, using mixture total covariance and Cholesky solves."""
    y = np.asarray(targets, dtype=np.float64)
    mean, covariance = prediction.mean, prediction.total_covariance
    if y.shape != mean.shape:
        raise ValueError("targets must match prediction means")
    values = np.empty(y.shape[0], dtype=np.float64)
    for index, (residual, matrix) in enumerate(zip(y - mean, covariance, strict=True)):
        stable = matrix + jitter * np.eye(matrix.shape[0])
        solution = np.linalg.solve(stable, residual)
        values[index] = residual @ solution / residual.shape[0]
    return np.maximum(values, jitter)


def fit_nrc_calibrator(prediction: GaussianMixturePrediction, targets: np.ndarray, distances: np.ndarray, *, ridge: float = 1e-6, scale_min: float = 0.25, scale_max: float = 4.0, jitter: float = 1e-8) -> NRCCalibrator:
    """Fit NRC-Cal by the documented closed-form ridge regression on calibration data."""
    if ridge < 0 or not (0 < scale_min <= scale_max):
        raise ValueError("Require ridge >= 0 and 0 < scale_min <= scale_max")
    d = np.asarray(distances, dtype=np.float64).reshape(-1)
    if d.shape[0] != prediction.means.shape[0]:
        raise ValueError("One NRC distance is required for each calibration target")
    mean, std = float(d.mean()), float(d.std())
    z = (d - mean) / max(std, jitter)
    design = np.column_stack((np.ones_like(z), z))
    penalty = np.diag((0.0, ridge))
    response = np.log(mahalanobis_scale_residuals(prediction, targets, jitter))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    return NRCCalibrator(float(coefficients[0]), float(coefficients[1]), mean, std, prediction.means.shape[2], ridge, scale_min, scale_max)


def select_ridge(prediction: GaussianMixturePrediction, targets: np.ndarray, distances: np.ndarray, candidates: tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2)) -> NRCCalibrator:
    """Select a deterministic calibration-only ridge value by univariate PCE or NLL."""
    from metrics.evaluation import negative_log_likelihood, probability_calibration_error

    best: tuple[float, NRCCalibrator] | None = None
    for ridge in candidates:
        fitted = fit_nrc_calibrator(prediction, targets, distances, ridge=ridge)
        calibrated = fitted.transform(prediction, distances)
        score = probability_calibration_error(calibrated, targets) if targets.shape[1] == 1 else negative_log_likelihood(calibrated, targets)
        if best is None or score < best[0]:
            best = (float(score), fitted)
    if best is None:
        raise RuntimeError("No ridge candidates supplied")
    return best[1]
