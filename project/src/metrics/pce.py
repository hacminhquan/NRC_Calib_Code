"""Probabilistic Calibration Error (PCE) and Quantile Recalibration (QR).

Formulas transcribed exactly from Dheur & Ben Taieb, "Probabilistic
Calibration by Design for Neural Network Regression" (AISTATS 2024) --
Section 2 ("Background on Probabilistic Calibration") -- whose full text
was already read in this project (it is the source repo notebooks 01-05
build on). No new formula-confirmation gate was needed for this module,
unlike `src/geometry/nrc.py`.

- Probability Integral Transform (PIT): Z = F_theta(Y | X).
- PCE(F_theta) = (1/M) * sum_j |alpha_j - Phi_EMP_theta(alpha_j)|, for M
  equidistant quantile levels 0 < alpha_1 < ... < alpha_M < 1 (paper fixes
  M=100). Phi_EMP_theta is the empirical CDF of the PIT values.
- Quantile Recalibration (Kuleshov et al., 2018, as restated in the QRT
  paper): F'_theta = Phi_Z o F_theta, where Phi_Z is the empirical CDF of
  PIT values Z'_i = F_theta(Y'_i | X'_i) from a separate calibration set.

This project's models (03/04) predict a `mixture_size=1` Gaussian for the
pilot, so F_theta(y|x) = Phi((y - mean(x)) / std(x)), the standard Normal
CDF -- implemented directly via `scipy.stats.norm`, not re-derived.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import norm


def compute_pit_gaussian(mean: np.ndarray, std: np.ndarray, y: np.ndarray) -> np.ndarray:
    """PIT values for a per-sample Normal(mean, std) predictive distribution.

    Z_i = F_theta(y_i | x_i) = Phi((y_i - mean_i) / std_i), the standard
    Normal CDF evaluated at the standardized residual.

    Parameters
    ----------
    mean, std:
        Per-sample predicted mean/std, shape [M] or [M, 1].
    y:
        True targets, same shape.

    Returns
    -------
    np.ndarray
        PIT values in [0, 1], shape [M].
    """
    mean, std, y = np.ravel(mean), np.ravel(std), np.ravel(y)
    if np.any(std <= 0):
        raise ValueError("std must be strictly positive for all samples.")
    return norm.cdf((y - mean) / std)


def empirical_cdf_eval(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Phi_EMP evaluated at `points`, fit from `reference`: fraction of
    `reference` values <= each point.

    Serves two roles in this module: (a) evaluating a PIT sample's own
    empirical CDF at the PCE quantile grid (`reference == points` is the
    array whose PCE is being measured), and (b) applying a *calibration-set*
    empirical CDF as the Quantile Recalibration map to a *different* (e.g.
    test-set) PIT sample -- see `apply_quantile_recalibration`.

    Parameters
    ----------
    reference:
        Sample defining the empirical CDF, shape [N].
    points:
        Points to evaluate that empirical CDF at, shape [M].

    Returns
    -------
    np.ndarray
        Phi_EMP(points), shape [M].
    """
    reference = np.ravel(reference)
    points = np.ravel(points)
    # For each point, fraction of reference <= point. Vectorized via searchsorted
    # on sorted reference (O((N+M) log N) instead of O(N*M)).
    sorted_ref = np.sort(reference)
    ranks = np.searchsorted(sorted_ref, points, side="right")
    return ranks / len(reference)


def compute_pce(z: np.ndarray, M: int = 100) -> float:
    """PCE(F_theta) = (1/M) * sum_j |alpha_j - Phi_EMP_theta(alpha_j)|.

    Source: QRT paper, Equation (2). `M` fixed at 100 there; kept as a
    parameter here but defaulting to the same value.

    Parameters
    ----------
    z:
        PIT values for the sample being evaluated (e.g. test-set PITs, or
        test-set PITs already passed through a recalibration map).
    M:
        Number of equidistant quantile levels in (0, 1).

    Returns
    -------
    float
        PCE value in [0, 1] (0 = perfectly calibrated on this sample).
    """
    alphas = np.linspace(0, 1, M + 2)[1:-1]  # M points strictly inside (0, 1), equidistant
    phi_emp = empirical_cdf_eval(z, alphas)
    return float(np.mean(np.abs(alphas - phi_emp)))


def apply_quantile_recalibration(z_calib: np.ndarray, z_target: np.ndarray) -> np.ndarray:
    """Quantile Recalibration: map `z_target` through the empirical CDF fit on `z_calib`.

    F'_theta = Phi_Z o F_theta, Phi_Z fit from a separate calibration set
    (source: QRT paper, "Quantile recalibration", Section 2). This is the
    Kuleshov et al. (2018) post-hoc recalibration map (QRT paper's "QRC" /
    `BASE + QR`, i.e. before QRT's own end-to-end training-time integration
    of this same map -- not implemented here, since this project's go/no-go
    only needs the plain post-hoc version to test against NRC-distance).

    Parameters
    ----------
    z_calib:
        PIT values on the calibration split (defines the recalibration map).
    z_target:
        PIT values to recalibrate (e.g. the test split's raw PITs).

    Returns
    -------
    np.ndarray
        Recalibrated PIT values, same shape as `z_target`.
    """
    return empirical_cdf_eval(z_calib, z_target)


def pce_before_and_after_qr(
    mean_calib: np.ndarray, std_calib: np.ndarray, y_calib: np.ndarray,
    mean_test: np.ndarray, std_test: np.ndarray, y_test: np.ndarray,
    M: int = 100,
) -> dict:
    """Convenience wrapper: PCE(BASE) and PCE(BASE + Quantile Recalibration) on test.

    Both computed from the *same* underlying model's predictions -- only
    the calibration map differs, matching how `06` needs to compare "raw"
    calibratability against "what QR already fixes" (mirroring how
    Calibration Bottleneck measures residual ECE after Temperature Scaling
    on the classification side).

    Returns
    -------
    dict
        ``{"pce_base": float, "pce_qrc": float, "z_test_raw": ndarray,
        "z_test_recalibrated": ndarray}``.
    """
    z_calib = compute_pit_gaussian(mean_calib, std_calib, y_calib)
    z_test = compute_pit_gaussian(mean_test, std_test, y_test)

    pce_base = compute_pce(z_test, M=M)

    z_test_recal = apply_quantile_recalibration(z_calib, z_test)
    pce_qrc = compute_pce(z_test_recal, M=M)

    return {
        "pce_base": pce_base, "pce_qrc": pce_qrc,
        "z_test_raw": z_test, "z_test_recalibrated": z_test_recal,
    }
