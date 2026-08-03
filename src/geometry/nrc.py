"""Exact published NRC metrics plus separately labelled NRC-Cal distances."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import minimize_scalar

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class NRCResult:
    """Published NRC metrics and the proposed consistent distance decomposition."""

    nrc1: float
    nrc2: float
    nrc3: float
    gamma: float | None
    residual_nrc1: np.ndarray
    residual_nrc2: np.ndarray
    sample_distance: np.ndarray
    dataset_distance: float
    weights: tuple[float, float, float]
    normalization: str


def _unit_rows(features: np.ndarray, centered: bool, epsilon: float) -> np.ndarray:
    """Normalize features exactly as either cited NRC paper specifies."""
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("features must have shape [examples, feature_dimension] with at least two examples")
    if centered:
        matrix = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= epsilon):
        raise ValueError("NRC normalization is undefined for a zero feature vector")
    return matrix / norms


def _orthonormal_columns(matrix: np.ndarray, rank: int, epsilon: float) -> np.ndarray:
    """Return a stable thin orthonormal basis for a column space."""
    basis, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    valid = int(np.sum(singular_values > epsilon))
    if valid < rank:
        raise ValueError(f"Required subspace rank {rank}, observed numerical rank {valid}")
    return basis[:, :rank]


def _projection_residuals(unit_features: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return squared residuals after orthogonal projection onto `basis`."""
    projected = (unit_features @ basis) @ basis.T
    return np.einsum("ij,ij->i", unit_features - projected, unit_features - projected)


def target_covariance(targets: np.ndarray) -> np.ndarray:
    """Compute the paper's `M^-1` empirical target covariance matrix."""
    y = np.asarray(targets, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2 or y.shape[0] < 2:
        raise ValueError("targets must have shape [examples, target_dimension]")
    centered = y - y.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / y.shape[0]
    return (covariance + covariance.T) / 2.0


def _published_nrc3(weight: np.ndarray, covariance: np.ndarray, epsilon: float) -> tuple[float, float | None]:
    """Compute NeurIPS-2024 NRC3 by bounded minimization over published gamma."""
    n = covariance.shape[0]
    if n == 1:
        return 0.0, None  # The source paper explicitly calls univariate NRC3 trivial.
    eigvals = np.linalg.eigvalsh(covariance)
    lambda_min = float(eigvals[0])
    if lambda_min <= epsilon:
        raise ValueError("Published NRC3 requires a full-rank positive-definite target covariance")
    sigma_half = np.asarray(sqrtm(covariance).real, dtype=np.float64)
    gram = weight @ weight.T
    gram_norm = np.linalg.norm(gram, ord="fro")
    if gram_norm <= epsilon:
        raise ValueError("Published normalized NRC3 is undefined for zero final-layer Gram matrix")

    def objective(gamma: float) -> float:
        target = sigma_half - np.sqrt(gamma) * np.eye(n)
        target_norm = np.linalg.norm(target, ord="fro")
        if target_norm <= epsilon:
            return float("inf")
        return float(np.linalg.norm(gram / gram_norm - target / target_norm, ord="fro") ** 2)

    upper = np.nextafter(lambda_min, 0.0)
    result = minimize_scalar(objective, bounds=(epsilon, upper), method="bounded")
    if not result.success:
        raise RuntimeError(f"NRC3 gamma minimization failed: {result.message}")
    return float(result.fun), float(result.x)


def compute_nrc(features: np.ndarray, targets: np.ndarray, mean_head_weight: np.ndarray, *, weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3), normalization: Literal["neurips_2024", "intrinsic_dimension_2025"] = "neurips_2024", epsilon: float = 1e-12) -> NRCResult:
    """Compute cited NRC1--3 and proposed NRC-Cal sample/dataset distances.

    NRC1--3 follow the cited definitions. `sample_distance` and
    `dataset_distance` are NRC-Cal proposal equations documented in
    `docs/methodology.md`, never published NRC metrics.
    """
    h = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    w = np.asarray(mean_head_weight, dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    if h.shape[0] != y.shape[0] or w.shape != (y.shape[1], h.shape[1]):
        raise ValueError("Expected features [M,d], targets [M,n], and mean_head_weight [n,d]")
    if np.any(np.asarray(weights) < 0) or not np.isclose(sum(weights), 1.0):
        raise ValueError("NRC-Cal weights must be nonnegative and sum to one")
    centered = normalization == "intrinsic_dimension_2025"
    unit = _unit_rows(h, centered, epsilon)
    n = y.shape[1]
    # PCA columns are right singular vectors of the examples-by-features matrix.
    _, _, right_vectors = np.linalg.svd(h - (h.mean(0, keepdims=True) if centered else 0.0), full_matrices=False)
    if right_vectors.shape[0] < n:
        raise ValueError("Feature dimension must be at least target dimension")
    pca_basis = right_vectors[:n].T
    weight_basis = _orthonormal_columns(w.T, n, epsilon)
    residual1 = _projection_residuals(unit, pca_basis)
    residual2 = _projection_residuals(unit, weight_basis)
    nrc1, nrc2 = float(residual1.mean()), float(residual2.mean())
    nrc3, gamma = _published_nrc3(w, target_covariance(y), epsilon)
    proposed_weights = np.asarray(weights, dtype=np.float64)
    if n == 1 and proposed_weights[2] != 0.0:
        proposed_weights[2] = 0.0
        proposed_weights /= proposed_weights.sum()
        LOGGER.info("Renormalized NRC-Cal weights because published univariate NRC3 is trivial")
    sample = proposed_weights[0] * residual1 + proposed_weights[1] * residual2 + proposed_weights[2] * nrc3
    dataset = proposed_weights[0] * nrc1 + proposed_weights[1] * nrc2 + proposed_weights[2] * nrc3
    if not np.isclose(sample.mean(), dataset, atol=1e-10):
        raise AssertionError("NRC-Cal sample/dataset consistency invariant failed")
    return NRCResult(nrc1, nrc2, nrc3, gamma, residual1, residual2, sample, float(dataset), tuple(proposed_weights), normalization)
