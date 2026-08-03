"""Paired statistical tests, multiple-testing correction, and rank summaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import friedmanchisquare, kendalltau, pearsonr, spearmanr, wilcoxon


@dataclass(frozen=True)
class CorrelationResult:
    """Correlation estimate, p-value, and nonparametric bootstrap interval."""

    name: str
    coefficient: float
    pvalue: float
    ci_low: float
    ci_high: float
    permutation_pvalue: float


def correlations(x: np.ndarray, y: np.ndarray, bootstrap_samples: int = 2_000, permutations: int = 5_000, seed: int = 0) -> tuple[CorrelationResult, ...]:
    """Compute Pearson/Spearman/Kendall with bootstrap CIs and permutation tests."""
    a, b = np.asarray(x, float).reshape(-1), np.asarray(y, float).reshape(-1)
    if a.size != b.size or a.size < 4:
        raise ValueError("Correlation requires equal arrays with at least four observations")
    rng = np.random.default_rng(seed)
    functions = (("pearson", lambda u, v: pearsonr(u, v).statistic, pearsonr), ("spearman", lambda u, v: spearmanr(u, v).statistic, spearmanr), ("kendall", lambda u, v: kendalltau(u, v).statistic, kendalltau))
    result: list[CorrelationResult] = []
    for name, coefficient_function, test_function in functions:
        observed, pvalue = test_function(a, b)
        draws = np.array([coefficient_function(a[index], b[index]) for index in rng.integers(0, a.size, (bootstrap_samples, a.size))])
        null = np.array([coefficient_function(a, rng.permutation(b)) for _ in range(permutations)])
        permutation_p = (1.0 + np.sum(np.abs(null) >= abs(observed))) / (permutations + 1.0)
        result.append(CorrelationResult(name, float(observed), float(pvalue), float(np.nanquantile(draws, .025)), float(np.nanquantile(draws, .975)), float(permutation_p)))
    return tuple(result)


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Return Holm-Bonferroni adjusted p-values in original order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (p.size - rank) * p[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def paired_statistics(scores: np.ndarray) -> dict[str, object]:
    """Run Friedman and pairwise Wilcoxon tests for `[datasets, methods]` scores."""
    matrix = np.asarray(scores, float)
    if matrix.ndim != 2 or matrix.shape[1] < 3:
        raise ValueError("Need scores for at least three methods")
    friedman = friedmanchisquare(*(matrix[:, column] for column in range(matrix.shape[1])))
    pairs: list[tuple[int, int, float]] = []
    for left in range(matrix.shape[1]):
        for right in range(left + 1, matrix.shape[1]):
            pairs.append((left, right, float(wilcoxon(matrix[:, left], matrix[:, right], zero_method="wilcox").pvalue)))
    return {"friedman_statistic": float(friedman.statistic), "friedman_pvalue": float(friedman.pvalue), "pairs": pairs, "holm_pvalues": holm_adjust(np.array([pair[2] for pair in pairs])).tolist(), "average_ranks": np.mean(np.argsort(np.argsort(matrix, axis=1), axis=1) + 1, axis=0).tolist()}
