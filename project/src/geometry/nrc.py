"""Neural Regression Collapse (NRC) geometry — notebook 05's formula-gated module.

Every formula below is transcribed exactly from Andriopoulos, Dong, Guo,
Zhao & Ross, "The Prevalence of Neural Collapse in Neural Multivariate
Regression" (NeurIPS 2024), Section 3.1 (definitions) and Appendix F
(closed-form gamma for NRC3), fetched and read in full before writing this
module. No formula here was guessed or approximated from the classification
NC1/NC2/NC3 analogy — every one is checked against the source equations
cited in each function's docstring.

Notation (matches the paper exactly):
- H = [h_1 ... h_M], h_i in R^d: penultimate-layer feature vectors.
- h~_i := h_i / ||h_i||_2: L2-normalized feature vectors.
- Y = [y_1 ... y_M], y_i in R^n: targets. Sigma = M^-1 (Y-Ybar)(Y-Ybar)^T,
  the n x n target covariance matrix.
- W: the n x d weight matrix of the network's linear prediction head
  (NOT the full output layer of a mixture-density network -- see
  `extract_mean_head_weight` for why only the mean sub-block is used here).
- proj(v | C): projection of v onto the subspace spanned by the columns of C.

A load-bearing, paper-stated fact this module respects rather than silently
overriding: for univariate regression (n=1), NRC3 is trivially zero/not
meaningful (paper, Appendix A.3: "we found NRC3 for univariate regression
to be not as meaningful, and therefore omitted the corresponding plots").
`compute_nrc3` returns `None` for n=1 rather than a fabricated number.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger("nrc_cal.geometry.nrc")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Shared linear-algebra primitives
# --------------------------------------------------------------------------- #


def normalize_rows(H: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize each row of H: h~_i = h_i / ||h_i||_2.

    Source: paper, Section 3.1, "the normalized feature vector h~_i := h_i . ||h_i||^-1".

    Parameters
    ----------
    H:
        Feature matrix, shape [M, d] (M samples, d-dimensional features).
    eps:
        Numerical floor to avoid division by zero for a (pathological)
        exactly-zero feature vector.

    Returns
    -------
    np.ndarray
        Row-normalized H, same shape.
    """
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(norms, eps)


def pca_components(H: np.ndarray, n: int) -> np.ndarray:
    """Top-n principal component directions of H (the paper's "H_PCAn").

    Standard PCA: center H, take the top-n eigenvectors of the (d x d)
    covariance matrix (equivalently, the top-n right singular vectors of
    centered H), by explained variance. Computed via SVD for numerical
    stability rather than an explicit covariance eigendecomposition.

    Source: paper, Section 3.1, "Let H_PCAn be the d x n matrix with the
    columns consisting of the n principal components of H."

    Parameters
    ----------
    H:
        Feature matrix, shape [M, d]. NOT pre-normalized -- PCA is defined
        on the paper's H, not H~ (see module docstring).
    n:
        Number of components (= target dimension in NRC theory).

    Returns
    -------
    np.ndarray
        [d, n] matrix with orthonormal columns.

    Raises
    ------
    ValueError
        If n exceeds min(M, d) (not enough samples/dimensions for n components).
    """
    M, d = H.shape
    if n > min(M, d):
        raise ValueError(f"Cannot compute {n} principal components from H of shape {H.shape}.")
    H_centered = H - H.mean(axis=0, keepdims=True)
    # SVD of centered data: columns of Vt.T (== V) are the principal directions,
    # ordered by decreasing singular value (== decreasing explained variance).
    _, _, Vt = np.linalg.svd(H_centered, full_matrices=False)
    return Vt[:n].T  # [d, n]


def project_onto_subspace(V: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Project each row of V onto the subspace spanned by the columns of C.

    General projection formula proj(v|C) = C (C^T C)^-1 C^T v, which reduces
    to C C^T v when C has orthonormal columns (as `pca_components` produces)
    but is implemented in the general form since `C = W^T` (used for NRC2)
    is not orthonormal in general.

    Source: paper, Section 3.1, "let proj(v|C) denote the projection of v
    onto the subspace spanned by the columns of C."

    Parameters
    ----------
    V:
        Vectors to project, shape [M, d] (one vector per row).
    C:
        Basis matrix, shape [d, n].

    Returns
    -------
    np.ndarray
        Projected vectors, shape [M, d].

    Raises
    ------
    numpy.linalg.LinAlgError
        If C^T C is singular (columns of C are linearly dependent) --
        e.g. a zero weight row for some target dimension.
    """
    CtC_inv = np.linalg.inv(C.T @ C)
    # proj_i = C (C^T C)^-1 C^T v_i, vectorized over rows of V:
    return V @ (C @ CtC_inv @ C.T).T


# --------------------------------------------------------------------------- #
# NRC1, NRC2
# --------------------------------------------------------------------------- #


def compute_nrc1(H: np.ndarray, n_target_dim: int) -> float:
    """NRC1: how far normalized features are from the top-n PCA subspace of H.

    Exact formula (paper, Section 3.1):
        NRC1 = (1/M) * sum_i || h~_i - proj(h~_i | H_PCAn) ||_2^2

    NRC1 -> 0 indicates the d-dimensional feature vectors collapse to a much
    lower n-dimensional subspace spanned by their own top-n principal
    components (paper's interpretation, same section).

    Parameters
    ----------
    H:
        Raw (non-normalized) feature matrix, shape [M, d].
    n_target_dim:
        n in the paper's notation -- the dimension of the regression target,
        NOT the mixture size (see `extract_mean_head_weight`).

    Returns
    -------
    float
        NRC1 value (>= 0; 0 would indicate perfect collapse).
    """
    H_pca = pca_components(H, n_target_dim)
    H_norm = normalize_rows(H)
    projected = project_onto_subspace(H_norm, H_pca)
    residual_sq_norms = np.sum((H_norm - projected) ** 2, axis=1)
    return float(np.mean(residual_sq_norms))


def compute_nrc2(H: np.ndarray, W: np.ndarray) -> float:
    """NRC2: how far normalized features are from the row space of W (self-duality).

    Exact formula (paper, Section 3.1):
        NRC2 = (1/M) * sum_i || h~_i - proj(h~_i | W^T) ||_2^2

    NRC2 -> 0 indicates self-duality: features also collapse to the
    n-dimensional space spanned by the rows of W (paper's interpretation,
    same section).

    Parameters
    ----------
    H:
        Raw (non-normalized) feature matrix, shape [M, d].
    W:
        The **mean-head** weight sub-block, shape [n, d] -- see
        `extract_mean_head_weight`. Using the full mixture-density output
        layer here (including std/mixture-logit rows) would not match the
        paper's "W" (the direct linear predictor of y).

    Returns
    -------
    float
        NRC2 value (>= 0; 0 would indicate perfect self-duality).
    """
    H_norm = normalize_rows(H)
    projected = project_onto_subspace(H_norm, W.T)
    residual_sq_norms = np.sum((H_norm - projected) ** 2, axis=1)
    return float(np.mean(residual_sq_norms))


# --------------------------------------------------------------------------- #
# Target covariance and NRC3
# --------------------------------------------------------------------------- #


def compute_target_covariance_sqrt(Y: np.ndarray) -> np.ndarray:
    """Sigma^{1/2}: the positive-definite square root of the target covariance.

    Source: paper, Section 3.1, Sigma := M^-1 (Y-Ybar)(Y-Ybar)^T; Section 4.1
    notes Sigma is assumed full rank / positive definite, hence has a unique
    positive-definite square root.

    For n=1 (paper, Appendix A.3), this reduces to a scalar: Sigma^{1/2} = sigma,
    the standard deviation of the (1-D) targets.

    Parameters
    ----------
    Y:
        Target matrix, shape [M, n].

    Returns
    -------
    np.ndarray
        [n, n] matrix (or a 1x1 array for n=1) -- Sigma^{1/2} via eigendecomposition.

    Raises
    ------
    ValueError
        If Sigma is not (numerically) positive definite -- e.g. a
        near-constant target dimension, or fewer samples than target
        dimensions, violating the paper's full-rank assumption.
    """
    n = Y.shape[1]
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    Sigma = (Y_centered.T @ Y_centered) / Y.shape[0]  # [n, n]

    if n == 1:
        std = float(np.sqrt(Sigma[0, 0]))
        if std <= 1e-8:
            raise ValueError("Target has ~zero variance; Sigma is not positive definite.")
        return np.array([[std]])

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    if np.min(eigvals) <= 1e-8:
        raise ValueError(
            f"Target covariance is not positive definite (min eigenvalue={np.min(eigvals):.2e}); "
            "the paper's full-rank assumption (Section 3.1) is violated -- check for a "
            "near-constant or collinear target dimension."
        )
    sqrt_eigvals = np.sqrt(np.maximum(eigvals, 0.0))
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def compute_nrc3(W: np.ndarray, Sigma_sqrt: np.ndarray, n_target_dim: int) -> Optional[float]:
    """NRC3: whether WW^T's Gram structure matches Sigma^{1/2} (up to the paper's gamma shift).

    Exact formulas (paper, Section 3.1 for the definition; Appendix F,
    Theorem F.1 for the closed-form optimal gamma used here instead of a
    grid search over gamma):

        NRC3(gamma) = || WW^T/||WW^T||_F - (Sigma^1/2 - gamma^1/2 I_n)/||Sigma^1/2 - gamma^1/2 I_n||_F ||_F^2

        gamma* = ( (tr(Sigma^1/2) - tr(WW^T)) / n )^2      [Theorem F.1,
                  valid when tr(Sigma^1/2) > tr(WW^T), which the theorem
                  proves gives a convex NRC3(gamma) with this unique minimum]

    Explicitly NOT computed for n=1: the paper states directly (Appendix
    A.3) that NRC3 is trivially zero and "not as meaningful" for univariate
    regression, and that the natural one-dimensional analogue is "also
    trivially true" -- they omit it from their own results for this reason.
    This function does the same rather than fabricating a number the
    source paper explicitly disowns.

    Parameters
    ----------
    W:
        The mean-head weight sub-block, shape [n, d].
    Sigma_sqrt:
        Output of `compute_target_covariance_sqrt`, shape [n, n].
    n_target_dim:
        n in the paper's notation.

    Returns
    -------
    Optional[float]
        NRC3 value for n_target_dim > 1; `None` for n_target_dim == 1
        (logged, not silently dropped).

    Raises
    ------
    ValueError
        If Theorem F.1's precondition tr(Sigma^1/2) > tr(WW^T) fails --
        the closed-form gamma* is not valid outside this regime, and this
        project does not fall back to a grid search (which the paper itself
        only used for the empirical Figure 2/4 curves, not as the primary
        method -- see Appendix F).
    """
    if n_target_dim == 1:
        logger.info(
            "NRC3 not computed for n=1 (univariate target): the source paper "
            "(Appendix A.3) states it is trivially zero and not meaningful in this case."
        )
        return None

    WWt = W @ W.T  # [n, n]
    tr_Sigma_sqrt = float(np.trace(Sigma_sqrt))
    tr_WWt = float(np.trace(WWt))

    if not (tr_Sigma_sqrt > tr_WWt):
        raise ValueError(
            f"Theorem F.1's precondition tr(Sigma^1/2)={tr_Sigma_sqrt:.4f} > "
            f"tr(WW^T)={tr_WWt:.4f} does not hold -- the closed-form gamma* is not "
            "valid here. This can happen for an undertrained or unusually-scaled "
            "model; re-check training before trusting NRC3 for this dataset."
        )

    gamma_star = ((tr_Sigma_sqrt - tr_WWt) / n_target_dim) ** 2
    shifted = Sigma_sqrt - np.sqrt(gamma_star) * np.eye(n_target_dim)

    WWt_norm = WWt / max(np.linalg.norm(WWt, ord="fro"), 1e-12)
    shifted_norm = shifted / max(np.linalg.norm(shifted, ord="fro"), 1e-12)

    diff = WWt_norm - shifted_norm
    return float(np.linalg.norm(diff, ord="fro") ** 2)


# --------------------------------------------------------------------------- #
# Bridging this project's mixture-density architecture to the paper's W
# --------------------------------------------------------------------------- #


def extract_mean_head_weight(output_layer_weight: np.ndarray, mixture_size: int) -> np.ndarray:
    """Slice out the mean-prediction rows of a MixturePrediction's output layer.

    Not from the source NRC papers (they study plain regression networks
    with a single linear head) -- this is this project's own, explicitly
    documented bridge to the mixture-density architecture actually used
    (03/04): `MixturePrediction`'s output layer produces
    `[means (K), rhos (K), mix_logits (K)]` stacked as one Linear layer
    (verified in `uq/models/general/mlp.py`, see `pilot_mixture_model.py`'s
    module docstring). NRC theory concerns the network's direct linear
    predictor of y -- for this architecture, that is only the mean rows;
    the std and mixture-logit rows are auxiliary outputs the theory does
    not model, and including them would not match the paper's "W".

    Parameters
    ----------
    output_layer_weight:
        The full output layer weight matrix, shape [3*mixture_size, d]
        (as produced by `MixturePrediction`'s internal `MLP.output_layer`).
    mixture_size:
        K. For the pilot's mixture_size=1, this returns a single [1, d] row
        -- exactly the "w" row vector of the paper's Section 4.2 univariate case.

    Returns
    -------
    np.ndarray
        [mixture_size, d] -- the mean-head rows only.

    Raises
    ------
    ValueError
        If `output_layer_weight`'s row count isn't a multiple of 3
        (a sign this isn't actually a MixturePrediction output layer, or
        upstream's output ordering has changed since this was verified).
    """
    total_rows = output_layer_weight.shape[0]
    if total_rows != 3 * mixture_size:
        raise ValueError(
            f"Expected output layer with 3*mixture_size={3*mixture_size} rows "
            f"(means, rhos, mix_logits), got {total_rows}. This does not match "
            "the verified MixturePrediction output structure -- do not proceed "
            "without re-checking uq/models/general/mlp.py."
        )
    return output_layer_weight[:mixture_size]  # first block == means, per verified source order
