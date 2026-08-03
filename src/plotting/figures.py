"""Scatter, heatmap, embedding, and critical-difference-style rank plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_correlation_scatter(frame: pd.DataFrame, x: str, y: str, path: str | Path) -> Path:
    """Save a labelled NRC-distance correlation scatter plot."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4)); sns.regplot(data=frame, x=x, y=y, ax=axis, scatter_kws={"s": 45})
    for row in frame.itertuples(): axis.annotate(str(row[1]), (getattr(row, x), getattr(row, y)), fontsize=7)
    figure.tight_layout(); figure.savefig(destination, dpi=300); plt.close(figure); return destination


def save_heatmap(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save a correlation heatmap from numeric result columns."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7, 5)); sns.heatmap(frame.corr(numeric_only=True), cmap="vlag", center=0, annot=True, fmt=".2f", ax=axis)
    figure.tight_layout(); figure.savefig(destination, dpi=300); plt.close(figure); return destination


def embedding(features: np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:
    """Return 2-D PCA or t-SNE coordinates; UMAP is optional at runtime."""
    matrix = np.asarray(features, float)
    if method == "pca": return PCA(n_components=2, random_state=seed).fit_transform(matrix)
    if method == "tsne": return TSNE(n_components=2, random_state=seed, init="pca").fit_transform(matrix)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
    raise ValueError("method must be pca, tsne, or umap")
