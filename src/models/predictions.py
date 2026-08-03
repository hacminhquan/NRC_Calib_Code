"""Validated Gaussian and Gaussian-mixture predictive distributions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm


@dataclass(frozen=True)
class GaussianMixturePrediction:
    """Batch of diagonal/full Gaussian mixtures with shape `[N,K,D]` parameters."""

    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self) -> None:
        weights, means, covariances = map(lambda a: np.asarray(a, dtype=np.float64), (self.weights, self.means, self.covariances))
        if means.ndim != 3 or weights.shape != means.shape[:2] or covariances.shape != means.shape[:2] + means.shape[2:] * 2:
            raise ValueError("Expected weights [N,K], means [N,K,D], covariances [N,K,D,D]")
        if np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1.0):
            raise ValueError("Mixture weights must be nonnegative and sum to one")
        if not np.allclose(covariances, np.swapaxes(covariances, -1, -2)):
            raise ValueError("Covariances must be symmetric")
        if np.any(np.linalg.eigvalsh(covariances) <= 0):
            raise ValueError("Covariances must be positive definite")

    @property
    def mean(self) -> np.ndarray:
        """Return mixture means with shape `[N,D]`."""
        return np.einsum("nk,nkd->nd", self.weights, self.means)

    @property
    def total_covariance(self) -> np.ndarray:
        """Return law-of-total-variance covariances with shape `[N,D,D]`."""
        mean = self.mean
        delta = self.means - mean[:, None, :]
        return np.einsum("nk,nkde->nde", self.weights, self.covariances) + np.einsum("nk,nkd,nke->nde", self.weights, delta, delta)

    def logpdf(self, targets: np.ndarray) -> np.ndarray:
        """Evaluate mixture log densities at `[N,D]` targets."""
        y = np.asarray(targets, dtype=np.float64)
        if y.shape != self.mean.shape:
            raise ValueError("targets must match prediction mean shape")
        terms = np.stack([np.log(self.weights[:, k]) + np.array([multivariate_normal.logpdf(y[i], self.means[i, k], self.covariances[i, k]) for i in range(y.shape[0])]) for k in range(self.weights.shape[1])], axis=1)
        return logsumexp(terms, axis=1)

    def cdf_1d(self, targets: np.ndarray) -> np.ndarray:
        """Evaluate 1-D mixture CDFs; multivariate CDF is intentionally unsupported."""
        if self.means.shape[2] != 1:
            raise ValueError("PIT/CDF evaluation is defined here only for univariate targets")
        y = np.asarray(targets, dtype=np.float64).reshape(-1)
        scales = np.sqrt(self.covariances[:, :, 0, 0])
        return np.sum(self.weights * norm.cdf((y[:, None] - self.means[:, :, 0]) / scales), axis=1)


def gaussian_prediction(mean: np.ndarray, variance: np.ndarray, minimum_variance: float = 1e-8) -> GaussianMixturePrediction:
    """Create a one-component diagonal Gaussian prediction from means/variances."""
    mu = np.asarray(mean, dtype=np.float64)
    var = np.maximum(np.asarray(variance, dtype=np.float64), minimum_variance)
    if mu.ndim == 1:
        mu, var = mu[:, None], var[:, None]
    if mu.shape != var.shape:
        raise ValueError("mean and variance must have matching shapes")
    covariance = np.eye(mu.shape[1])[None, None] * var[:, None, :, None]
    return GaussianMixturePrediction(np.ones((mu.shape[0], 1)), mu[:, None, :], covariance)
