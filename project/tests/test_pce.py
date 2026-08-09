"""Unit tests for src.metrics.pce."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import pce  # noqa: E402


# --------------------------------------------------------------------------- #
# compute_pit_gaussian
# --------------------------------------------------------------------------- #


def test_pit_is_half_when_y_equals_mean():
    mean = np.array([0.0, 5.0, -3.0])
    std = np.array([1.0, 2.0, 0.5])
    y = mean.copy()
    z = pce.compute_pit_gaussian(mean, std, y)
    np.testing.assert_allclose(z, 0.5, atol=1e-10)


def test_pit_in_unit_interval_for_arbitrary_inputs():
    rng = np.random.RandomState(0)
    mean, std = rng.randn(200), rng.uniform(0.1, 5, 200)
    y = rng.randn(200)
    z = pce.compute_pit_gaussian(mean, std, y)
    assert np.all((z >= 0) & (z <= 1))


def test_pit_raises_on_nonpositive_std():
    with pytest.raises(ValueError):
        pce.compute_pit_gaussian(np.array([0.0]), np.array([0.0]), np.array([1.0]))


def test_pit_matches_hand_computed_normal_cdf():
    # y one std above the mean -> Phi(1) ~ 0.8413
    z = pce.compute_pit_gaussian(np.array([0.0]), np.array([1.0]), np.array([1.0]))
    np.testing.assert_allclose(z[0], 0.8413, atol=1e-4)


# --------------------------------------------------------------------------- #
# empirical_cdf_eval
# --------------------------------------------------------------------------- #


def test_empirical_cdf_eval_matches_hand_computation():
    reference = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    points = np.array([0.25, 0.5, 0.05])
    result = pce.empirical_cdf_eval(reference, points)
    # <=0.25: {0.1,0.2} -> 2/5=0.4; <=0.5: all 5 -> 1.0; <=0.05: none -> 0.0
    np.testing.assert_allclose(result, [0.4, 1.0, 0.0])


def test_empirical_cdf_eval_self_gives_rank_based_cdf():
    reference = np.array([1.0, 2.0, 3.0])
    result = pce.empirical_cdf_eval(reference, reference)
    np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1.0])


# --------------------------------------------------------------------------- #
# compute_pce
# --------------------------------------------------------------------------- #


def test_pce_near_zero_for_perfectly_uniform_pits():
    # A dense, exactly-evenly-spaced grid in (0,1) is as close to "PIT ~ Uniform(0,1)" as
    # a finite sample gets.
    z = np.linspace(0.0005, 0.9995, 2000)
    val = pce.compute_pce(z, M=100)
    assert val < 0.01


def test_pce_high_for_degenerate_pits():
    # All PITs identical (e.g. model always predicts the same relative
    # position) is about as miscalibrated as it gets.
    z = np.full(200, 0.5)
    val = pce.compute_pce(z, M=100)
    assert val > 0.2


def test_pce_is_nonnegative_and_bounded():
    rng = np.random.RandomState(1)
    z = rng.uniform(0, 1, 500)
    val = pce.compute_pce(z)
    assert 0 <= val <= 1


# --------------------------------------------------------------------------- #
# apply_quantile_recalibration
# --------------------------------------------------------------------------- #


def test_recalibration_of_calib_set_against_itself_is_uniform_ranked():
    rng = np.random.RandomState(2)
    z_calib = rng.beta(2, 5, 300)  # deliberately non-uniform
    z_recal = pce.apply_quantile_recalibration(z_calib, z_calib)
    # Recalibrating the calibration set against itself must reproduce its
    # own rank-based empirical CDF -- i.e. become (approximately) uniform.
    assert pce.compute_pce(z_recal) < pce.compute_pce(z_calib)


def test_recalibration_reduces_pce_on_held_out_miscalibrated_model():
    """The key integration check: a deliberately underconfident model
    (predicted std too large relative to the true noise) should show high
    PCE before QR and lower PCE after -- QR should actually help, not just
    run without crashing."""
    rng = np.random.RandomState(3)
    n_calib, n_test = 2000, 2000
    true_std = 1.0
    predicted_std = 3.0  # badly overestimated -> underconfident predictive distribution

    mean_calib = np.zeros(n_calib)
    y_calib = rng.randn(n_calib) * true_std
    mean_test = np.zeros(n_test)
    y_test = rng.randn(n_test) * true_std

    result = pce.pce_before_and_after_qr(
        mean_calib, np.full(n_calib, predicted_std), y_calib,
        mean_test, np.full(n_test, predicted_std), y_test,
    )
    assert result["pce_qrc"] < result["pce_base"]
    assert result["pce_base"] > 0.03  # meaningfully miscalibrated to begin with


def test_pce_before_and_after_qr_returns_expected_keys():
    rng = np.random.RandomState(4)
    n = 100
    result = pce.pce_before_and_after_qr(
        np.zeros(n), np.ones(n), rng.randn(n),
        np.zeros(n), np.ones(n), rng.randn(n),
    )
    assert set(result.keys()) == {"pce_base", "pce_qrc", "z_test_raw", "z_test_recalibrated"}
