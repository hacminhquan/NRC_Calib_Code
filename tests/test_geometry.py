"""Unit tests for exact NRC equations and proposed distance invariants."""
from __future__ import annotations

import numpy as np
import pytest

from geometry.nrc import compute_nrc, target_covariance


def test_target_covariance_uses_population_denominator() -> None:
    targets = np.array([[0.0], [1.0], [2.0]])
    assert np.allclose(target_covariance(targets), [[2 / 3]])


def test_nrc_requires_matching_final_head() -> None:
    with pytest.raises(ValueError):
        compute_nrc(np.ones((4, 3)), np.ones((4, 1)), np.ones((2, 3)))


def test_univariate_nrc3_is_published_trivial_zero() -> None:
    rng = np.random.default_rng(2)
    result = compute_nrc(rng.normal(size=(20, 3)), rng.normal(size=(20, 1)), rng.normal(size=(1, 3)), weights=(0.5, 0.5, 0.0))
    assert result.nrc3 == 0.0 and result.gamma is None
    assert np.isclose(result.sample_distance.mean(), result.dataset_distance)


def test_multivariate_distance_consistency() -> None:
    rng = np.random.default_rng(4)
    result = compute_nrc(rng.normal(size=(40, 5)), rng.normal(size=(40, 2)), rng.normal(size=(2, 5)), weights=(1 / 3, 1 / 3, 1 / 3))
    assert np.isfinite(result.nrc3)
    assert np.isclose(result.sample_distance.mean(), result.dataset_distance)
