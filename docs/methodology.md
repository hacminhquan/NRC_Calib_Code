# NRC-Cal Methodology and Attribution

This document is the normative methods specification. Equations labelled
**Published NRC theory** are transcribed from Andriopoulos et al., *The
Prevalence of Neural Collapse in Neural Multivariate Regression*, NeurIPS
2024. Equations labelled **NRC-Cal proposal** are original methodology in this
repository, not results or claims of either NRC paper.

## Notation

For `M` examples, `x_i` is input `i`, `y_i in R^n` its target, and
`h_i in R^d` its last-hidden-layer feature. `H=[h_1,...,h_M] in R^(d x M)`.
`W in R^(n x d)` is the final affine mean-head weight matrix. `||.||_2` and
`||.||_F` are Euclidean and Frobenius norms. `P_C(v)` denotes orthogonal
projection of `v` onto the column span of `C`. `Sigma` is the population-form
empirical target covariance `M^-1 (Y-Ybar)(Y-Ybar)^T`, and `lambda_min` is
its smallest eigenvalue. `I_n` is the `n x n` identity matrix.

## Published NRC theory (verbatim mathematical content)

The NeurIPS 2024 paper defines `tilde(h_i)=h_i ||h_i||_2^-1` and lets
`H_PCA_n` contain the first `n` principal components of `H`. NRC emerges when

```
NRC1 = M^-1 sum_i ||tilde(h_i) - P_{H_PCA_n}(tilde(h_i))||_2^2 -> 0,
NRC2 = M^-1 sum_i ||tilde(h_i) - P_{W^T}(tilde(h_i))||_2^2 -> 0,
NRC3 = || WW^T/||WW^T||_F
       - (Sigma^(1/2)-gamma^(1/2)I_n)/||Sigma^(1/2)-gamma^(1/2)I_n||_F ||_F^2 -> 0,
```

for a constant `gamma in (0, lambda_min)`. The paper specifies that in the
univariate case NRC3 is trivially zero. It finds `gamma` by minimizing NRC3
for the final trained weight matrix. The 2025 intrinsic-dimension paper uses
a distinct *centered-normalized* NRC1 variant, with
`tilde(h_i)=(h_i-hbar)/||h_i-hbar||_2`; it is exposed as an explicitly named
alternative, never silently substituted for the 2024 definition.

Source snapshots: `references/nrc_neurips_2024/neurips_2024.tex:243-285` and
`references/nrc_intrinsic_dimension_2025/neurips_2026.tex:253-278`.

## NRC-Cal proposal: distances

**New assumption A1 (frozen representation).** Feature vectors and final
mean-head weights are measured after training and remain fixed. **A2
(nonzero features).** Every retained feature has norm greater than epsilon.

Define published per-example residuals
`r1_i=||tilde(h_i)-P_{H_PCA_n}(tilde(h_i))||_2^2` and
`r2_i=||tilde(h_i)-P_{W^T}(tilde(h_i))||_2^2`. Let `m1=M^-1 sum r1_i`,
`m2=M^-1 sum r2_i`, and `m3=NRC3`. Let `w=(w1,w2,w3)` satisfy `w_j>=0` and
`sum_j w_j=1`. For univariate targets we set `w3=0` and renormalize `(w1,w2)`
because the published paper says NRC3 is trivial.

**Dataset-level NRC-distance (new):**
`D_dataset = w1 m1 + w2 m2 + w3 m3`.

**Sample-level NRC-distance (new):**
`D_i = w1 r1_i + w2 r2_i + w3 m3`.

Motivation: each term is an existing bounded squared geometric residual;
convex aggregation adds no uncalibrated scale or learned representation.
The key property is exact consistency: `M^-1 sum_i D_i = D_dataset`.
With normalized inputs, `r1_i,r2_i in [0,1]` and `m3 in [0,4]`, hence
`D_i,D_dataset in [0,4]`. PCA costs `O(M d min(M,d))`; projection residuals
cost `O(Mdn)` using thin orthonormal bases; NRC3 costs `O(n^3+K n^2)` for `K`
candidate gamma evaluations. Expected behavior: large values flag departure
from the three collapse relationships, not calibration error itself.

## NRC-Cal proposal: closed-form calibration map

Let an arbitrary frozen Gaussian or `K`-component Gaussian mixture yield mean
`mu_i in R^n` and total covariance `V_i` (strictly positive definite). Define
the squared Mahalanobis residual `q_i=(y_i-mu_i)^T V_i^-1(y_i-mu_i)/n` and
the standardized geometry covariate `z_i=(D_i-Dbar)/(s_D+epsilon)`, where
`Dbar=M^-1 sum_i D_i` and `s_D=[M^-1 sum_i(D_i-Dbar)^2]^(1/2)`.

**New assumption A3 (log-scale model).** On an independent calibration split,
`log(q_i)=a+b z_i+eta_i`, with finite-variance, mean-zero residual `eta_i`.
For an exactly calibrated Gaussian, `n q_i` is chi-square with `n` degrees of
freedom. Its expected log scale is
`tau_n=psi(n/2)+log(2)-log(n)`, where `psi` is the digamma function.

With `X=[1,z]`, define the ridge-stabilized, closed-form estimate
`theta=(a,b)^T=(X^T X + lambda diag(0,1))^-1 X^T log(q)`. Here `lambda>=0`
is a fixed stability penalty, chosen by a deterministic calibration-only grid
search minimizing PCE; `lambda=0` is ordinary least squares whenever the
matrix is nonsingular. The correction is
`s_i=clip(exp((a+b z_i-tau_n)/2), s_min, s_max)`.

For a Gaussian, NRC-Cal maps `(mu_i,V_i)` to `(mu_i,s_i^2 V_i)`. For a
mixture with component weights `pi_ik`, means `mu_ik`, and covariances
`V_ik`, write `mu_i=sum_k pi_ik mu_ik` and map
`mu'_ik=mu_i+s_i(mu_ik-mu_i)`, `V'_ik=s_i^2 V_ik`, preserving `pi_ik`.
This is a valid Gaussian mixture and exactly maps its total covariance to
`s_i^2 V_i`; therefore it works without retraining or changing mixture mass.

**Proposition (new; assumptions A1--A3 and no clipping).** The map makes the
fitted conditional log-Mahalanobis residual independent of `z` in the linear
model: `log(q_i/s_i^2)=tau_n+eta_i`. Proof: substitute the definition of
`s_i^2` into `log q_i - log s_i^2`. Thus the fitted geometry-dependent scale
trend is removed while location and mixture weights remain unchanged.

**Stability.** Ridge makes `X^T X + lambda diag(0,1)` invertible whenever the
intercept column is present and `lambda>0`; clipping gives
`s_i in [s_min,s_max]`, so transformed covariance eigenvalues lie in
`[s_min^2 lambda_min(V_i), s_max^2 lambda_max(V_i)]`. Fit complexity is
`O(M)` for a two-column regression after `q`; Gaussian evaluation is
`O(M n^3)` with Cholesky solves and mixture moments are `O(M K n^2)`.

## Algorithms and ablations

**Algorithm 1 (new NRC-Cal distance, pseudocode).** Input: frozen `H,Y,W` and
convex `w`. (1) Normalize each feature exactly according to the selected
cited paper. (2) Compute `r1_i` by projection onto top-`n` PCA columns and
`r2_i` by projection onto `col(W^T)`. (3) Minimize the published NRC3
objective over `gamma in (0,lambda_min)`; set it to zero only for `n=1`, as
specified by the source. (4) Renormalize `w1,w2` when `n=1`. (5) Return
`D_i=w1 r1_i+w2 r2_i+w3 NRC3` and its mean `D_dataset`.

**Algorithm 2 (new NRC-Cal map, pseudocode).** Input: calibration targets,
frozen Gaussian/mixture prediction, and calibration `D_i`. (1) Calculate
total mixture covariance and `q_i`. (2) Standardize `D_i` into `z_i`. (3)
For each fixed ridge candidate, solve the displayed two-by-two normal
equations and select the calibration-only PCE minimizer. (4) Compute clipped
positive scales. (5) At inference, transform every mixture component with
the displayed mean/covariance equations, then evaluate unchanged frozen
means and transformed uncertainty.

The code implementations are `src/geometry/nrc.py` and
`src/calibration/nrc_cal.py`; their unit tests verify the mean-distance and
mixture-mean/positive-definiteness invariants.

Ablations are: `D=NRC1`, `D=NRC2`, `D=NRC3` (multivariate only), equal-weight
full distance, no geometry (`z=0`, global scale), no ridge, and no clipping.
All ablations retain the same frozen features, calibration split, and metrics.
