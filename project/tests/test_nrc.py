"""Unit tests for src.geometry.nrc.

The strongest test here (`test_theorem_4_1_construction_gives_near_zero_nrc`)
does not just check "reasonable-looking" behavior -- it constructs H, W, Y
using the *exact* closed-form global-minimum solution given by the source
paper's own Theorem 4.1 / Corollary 4.2, and checks that NRC1, NRC2, NRC3 as
implemented here all correctly go to ~0 on that construction. If the
formulas here were subtly wrong, this test -- derived independently from
the paper's math, not from this module's own code -- would very likely catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import nrc  # noqa: E402


# --------------------------------------------------------------------------- #
# normalize_rows
# --------------------------------------------------------------------------- #


def test_normalize_rows_gives_unit_norm():
    H = np.random.RandomState(0).randn(50, 8) * 5
    H_norm = nrc.normalize_rows(H)
    norms = np.linalg.norm(H_norm, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_normalize_rows_preserves_direction():
    H = np.array([[3.0, 4.0]])  # norm 5
    H_norm = nrc.normalize_rows(H)
    np.testing.assert_allclose(H_norm, [[0.6, 0.8]])


# --------------------------------------------------------------------------- #
# pca_components
# --------------------------------------------------------------------------- #


def test_pca_components_are_orthonormal():
    H = np.random.RandomState(1).randn(200, 10)
    P = nrc.pca_components(H, n=3)
    assert P.shape == (10, 3)
    gram = P.T @ P
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-6)


def test_pca_components_recover_known_high_variance_axes():
    """Data with huge variance along e0,e1 and tiny variance elsewhere:
    top-2 PCA directions must span {e0, e1}."""
    rng = np.random.RandomState(2)
    M, d = 500, 6
    H = rng.randn(M, d) * 0.01
    H[:, 0] += rng.randn(M) * 100
    H[:, 1] += rng.randn(M) * 50
    P = nrc.pca_components(H, n=2)
    # The subspace spanned by P should be ~span{e0, e1}: projecting e0, e1
    # onto span(P) should recover them almost exactly.
    for axis in (0, 1):
        e = np.zeros((1, d)); e[0, axis] = 1.0
        proj = nrc.project_onto_subspace(e, P)
        np.testing.assert_allclose(proj, e, atol=1e-2)


def test_pca_components_raises_when_n_too_large():
    H = np.random.RandomState(0).randn(5, 3)
    with pytest.raises(ValueError):
        nrc.pca_components(H, n=10)


# --------------------------------------------------------------------------- #
# project_onto_subspace
# --------------------------------------------------------------------------- #


def test_projection_is_idempotent():
    rng = np.random.RandomState(3)
    C = rng.randn(10, 3)
    V = rng.randn(20, 10)
    once = nrc.project_onto_subspace(V, C)
    twice = nrc.project_onto_subspace(once, C)
    np.testing.assert_allclose(once, twice, atol=1e-6)


def test_projection_onto_full_space_is_identity():
    rng = np.random.RandomState(4)
    d = 5
    C = np.eye(d)  # full-rank, spans all of R^d
    V = rng.randn(15, d)
    proj = nrc.project_onto_subspace(V, C)
    np.testing.assert_allclose(proj, V, atol=1e-6)


def test_projection_onto_orthogonal_complement_gives_zero_for_own_axis():
    C = np.array([[0.0], [1.0], [0.0]])  # spans only e1
    v = np.array([[1.0, 0.0, 0.0]])  # orthogonal to C's span
    proj = nrc.project_onto_subspace(v, C)
    np.testing.assert_allclose(proj, [[0.0, 0.0, 0.0]], atol=1e-8)


# --------------------------------------------------------------------------- #
# compute_nrc1 / compute_nrc2 -- basic collapsed vs. non-collapsed sanity
# --------------------------------------------------------------------------- #


def test_nrc1_near_zero_for_perfectly_collapsed_features():
    """Features constructed to lie exactly in a 2-D subspace of R^10."""
    rng = np.random.RandomState(5)
    M, d, n = 300, 10, 2
    coeffs = rng.randn(M, n)
    basis = rng.randn(d, n)
    basis, _ = np.linalg.qr(basis)  # orthonormalize for a clean construction
    H = coeffs @ basis.T  # exactly rank-n
    val = nrc.compute_nrc1(H, n_target_dim=n)
    assert val < 1e-6


def test_nrc1_bounded_away_from_zero_for_isotropic_features():
    """Isotropic (no true low-rank structure) features should NOT collapse."""
    rng = np.random.RandomState(6)
    H = rng.randn(500, 20)  # full-rank isotropic noise
    val = nrc.compute_nrc1(H, n_target_dim=2)
    assert val > 0.1  # nowhere near collapsed


def test_nrc2_near_zero_when_features_align_with_W():
    rng = np.random.RandomState(7)
    M, d, n = 300, 10, 2
    W = rng.randn(n, d)
    coeffs = rng.randn(M, n)
    H = coeffs @ W  # H lies exactly in span(W^T) by construction
    val = nrc.compute_nrc2(H, W)
    assert val < 1e-6


def test_nrc2_bounded_away_from_zero_when_misaligned():
    rng = np.random.RandomState(8)
    M, d, n = 300, 10, 2
    W = np.zeros((n, d)); W[0, 0] = 1.0; W[1, 1] = 1.0  # spans e0, e1 only
    H = rng.randn(M, d)  # generic, not confined to span(e0, e1)
    H[:, :2] *= 0.001  # explicitly suppress the e0,e1 components so H lives elsewhere
    val = nrc.compute_nrc2(H, W)
    assert val > 0.1


# --------------------------------------------------------------------------- #
# compute_target_covariance_sqrt
# --------------------------------------------------------------------------- #


def test_covariance_sqrt_squares_back_to_covariance():
    rng = np.random.RandomState(9)
    M, n = 1000, 3
    A = rng.randn(n, n)
    true_cov = A @ A.T + np.eye(n) * 0.1  # ensure PD
    L = np.linalg.cholesky(true_cov)
    Y = rng.randn(M, n) @ L.T
    Sigma_sqrt = nrc.compute_target_covariance_sqrt(Y)
    reconstructed = Sigma_sqrt @ Sigma_sqrt
    empirical_cov = np.cov(Y.T, bias=True)
    np.testing.assert_allclose(reconstructed, empirical_cov, atol=1e-6)


def test_covariance_sqrt_univariate_is_std():
    rng = np.random.RandomState(10)
    Y = (rng.randn(1000, 1) * 3.0) + 5.0
    Sigma_sqrt = nrc.compute_target_covariance_sqrt(Y)
    assert Sigma_sqrt.shape == (1, 1)
    np.testing.assert_allclose(Sigma_sqrt[0, 0], Y.std(), atol=1e-6)


def test_covariance_sqrt_raises_on_constant_target():
    Y = np.ones((100, 1)) * 7.0  # zero variance
    with pytest.raises(ValueError):
        nrc.compute_target_covariance_sqrt(Y)


# --------------------------------------------------------------------------- #
# compute_nrc3
# --------------------------------------------------------------------------- #


def test_nrc3_returns_none_for_univariate():
    W = np.array([[1.0, 2.0, 3.0]])
    Sigma_sqrt = np.array([[2.0]])
    result = nrc.compute_nrc3(W, Sigma_sqrt, n_target_dim=1)
    assert result is None


def test_nrc3_raises_when_theorem_precondition_violated():
    # tr(Sigma^1/2) very small, tr(WW^T) very large -> precondition tr(Sigma^1/2) > tr(WW^T) fails
    W = np.eye(2) * 100
    Sigma_sqrt = np.eye(2) * 0.01
    with pytest.raises(ValueError):
        nrc.compute_nrc3(W, Sigma_sqrt, n_target_dim=2)


# --------------------------------------------------------------------------- #
# THE key end-to-end test: paper's own Theorem 4.1 / Corollary 4.2 construction
# --------------------------------------------------------------------------- #


def test_theorem_4_1_construction_gives_near_zero_nrc():
    """Construct H, W, Y using the paper's own closed-form global-minimum
    solution (Theorem 4.1, uncorrelated-target case worked in Appendix D.1)
    and verify NRC1, NRC2, NRC3 (as implemented here, independently) all
    correctly evaluate to ~0 on it.

    Setup (paper notation): n=2, uncorrelated targets with variances
    sigma1^2=4, sigma2^2=1 (so Sigma=diag(4,1), Sigma^1/2=diag(2,1)),
    c=0.25 (0 < c < sigma_min^2=1, matching Theorem 4.1 Case I),
    lambda_H = lambda_W = 1 for simplicity.
    """
    rng = np.random.RandomState(42)
    d, n, M = 8, 2, 2000

    sigma1_sq, sigma2_sq = 4.0, 1.0
    c = 0.25
    Sigma = np.diag([sigma1_sq, sigma2_sq])
    Sigma_sqrt_true = np.diag([np.sqrt(sigma1_sq), np.sqrt(sigma2_sq)])  # diag(2, 1)
    Sigma_inv_sqrt = np.diag([1 / np.sqrt(sigma1_sq), 1 / np.sqrt(sigma2_sq)])

    # --- Construct Y with EXACTLY covariance Sigma (whitened-then-colored,
    # avoids sampling noise so this is an exact, not approximate, check) ---
    Y_raw = rng.randn(M, n)
    raw_cov = np.cov(Y_raw.T, bias=True)
    raw_cov_inv_sqrt = np.linalg.inv(np.linalg.cholesky(raw_cov))
    Y_white = Y_raw @ raw_cov_inv_sqrt.T  # exactly identity covariance
    Y = Y_white @ Sigma_sqrt_true.T + np.array([10.0, -3.0])  # exact cov=Sigma, arbitrary mean
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    empirical_cov = np.cov(Y.T, bias=True)
    np.testing.assert_allclose(empirical_cov, Sigma, atol=1e-6)  # construction sanity check

    # --- Corollary 4.2 construction: A^{1/2} = diag(sqrt(sigma_j - sqrt(c))) ---
    A_sqrt = np.diag([np.sqrt(np.sqrt(sigma1_sq) - np.sqrt(c)), np.sqrt(np.sqrt(sigma2_sq) - np.sqrt(c))])
    # W = sqrt(lambda_H/lambda_W) * A^{1/2} * R, with lambda_H=lambda_W=1 and
    # R = first n rows of I_d (trivially semi-orthogonal: R R^T = I_n).
    W_true = np.zeros((n, d))
    W_true[:, :n] = A_sqrt  # [n, d], nonzero only in first n columns

    # H (paper, d x M) = sqrt(lambda_W/lambda_H) * W^T * Sigma^{-1/2} * (Y-Ybar)^T
    # In row-major [M, d] convention used by this module:
    H = Y_centered @ Sigma_inv_sqrt @ W_true  # [M, n] @ [n, n] @ [n, d] -> [M, d]

    # --- Verify NRC1, NRC2, NRC3 all ~0 on this theoretically-optimal construction ---
    nrc1 = nrc.compute_nrc1(H, n_target_dim=n)
    nrc2 = nrc.compute_nrc2(H, W_true)
    Sigma_sqrt_computed = nrc.compute_target_covariance_sqrt(Y)
    nrc3 = nrc.compute_nrc3(W_true, Sigma_sqrt_computed, n_target_dim=n)

    assert nrc1 < 1e-6, f"NRC1={nrc1} should be ~0 on the paper's own optimal construction"
    assert nrc2 < 1e-6, f"NRC2={nrc2} should be ~0 on the paper's own optimal construction"
    assert nrc3 is not None and nrc3 < 1e-6, f"NRC3={nrc3} should be ~0 on the paper's own optimal construction"

    # Bonus check: our closed-form gamma* should recover c=0.25 exactly
    # (derived by hand in this test's docstring context: tr(Sigma^1/2)-tr(WW^T) = sqrt(c)*n).
    WWt = W_true @ W_true.T
    gamma_star = ((np.trace(Sigma_sqrt_computed) - np.trace(WWt)) / n) ** 2
    assert abs(gamma_star - c) < 1e-6, f"gamma*={gamma_star} should recover c={c} exactly"


# --------------------------------------------------------------------------- #
# extract_mean_head_weight
# --------------------------------------------------------------------------- #


def test_extract_mean_head_weight_slices_first_block():
    K, d = 3, 5
    W_full = np.arange(3 * K * d).reshape(3 * K, d).astype(float)
    means = nrc.extract_mean_head_weight(W_full, mixture_size=K)
    np.testing.assert_array_equal(means, W_full[:K])


def test_extract_mean_head_weight_single_gaussian_gives_one_row():
    W_full = np.arange(3 * 5).reshape(3, 5).astype(float)  # mixture_size=1 -> 3 rows total
    means = nrc.extract_mean_head_weight(W_full, mixture_size=1)
    assert means.shape == (1, 5)
    np.testing.assert_array_equal(means, W_full[:1])


def test_extract_mean_head_weight_raises_on_wrong_row_count():
    W_full = np.zeros((7, 4))  # not a multiple of 3
    with pytest.raises(ValueError):
        nrc.extract_mean_head_weight(W_full, mixture_size=2)
