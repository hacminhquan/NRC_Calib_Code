"""Adapters from common frozen Gaussian head tensors to validated predictions."""
from __future__ import annotations

import numpy as np

from models.predictions import GaussianMixturePrediction, gaussian_prediction


def prediction_from_heads(means: np.ndarray, log_variances: np.ndarray, logits: np.ndarray | None = None, minimum_variance: float = 1e-8) -> GaussianMixturePrediction:
    """Build Gaussian, generic mixture, Mixture-3, or Mixture-10 predictions.

    `means` and `log_variances` accept `[N,D]` for a Gaussian or `[N,K,D]`
    for any mixture count `K`, including 3 and 10. Optional logits have
    shape `[N,K]` and are normalized with a stable softmax.
    """
    mean = np.asarray(means, dtype=np.float64)
    variance = np.maximum(np.exp(np.asarray(log_variances, dtype=np.float64)), minimum_variance)
    if mean.ndim == 2:
        return gaussian_prediction(mean, variance, minimum_variance)
    if mean.ndim != 3 or mean.shape != variance.shape:
        raise ValueError("Expected matching [N,D] or [N,K,D] mean/log-variance tensors")
    n_examples, components, dimension = mean.shape
    if logits is None:
        weights = np.full((n_examples, components), 1.0 / components)
    else:
        score = np.asarray(logits, dtype=np.float64)
        if score.shape != (n_examples, components):
            raise ValueError("logits must have shape [N,K]")
        score -= score.max(axis=1, keepdims=True)
        weights = np.exp(score); weights /= weights.sum(axis=1, keepdims=True)
    covariance = np.eye(dimension)[None, None] * variance[:, :, :, None]
    return GaussianMixturePrediction(weights, mean, covariance)
