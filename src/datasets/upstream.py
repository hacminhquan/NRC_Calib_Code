"""Exact QRT upstream data acquisition and split-cache construction."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from datasets.manifest import qrt57_manifest
from datasets.splits import TabularSplits
from utils.io import save_arrays


def qrt_split(x: np.ndarray, y: np.ndarray, seed: int = 0) -> TabularSplits:
    """Reproduce QRT's `[.65,0,.10,.15,.10]` random-split allocation exactly."""
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching first dimension")
    size = x.shape[0]
    lengths = (np.array([.65, 0., .10, .15, .10]) * size).astype(int)
    overflow = max(0, int(lengths[3] - 2048))
    lengths[3] -= overflow
    mask = (lengths != 0) & (np.arange(5) != 3)
    lengths[mask] += int(overflow / int(mask.sum()))
    lengths[-1] = size - lengths[:-1].sum()
    indices = torch.randperm(size, generator=torch.Generator().manual_seed(seed)).numpy()
    train, inter, validation, calibration, test = np.split(indices, np.cumsum(lengths)[:-1])
    if inter.size:
        raise AssertionError("QRT baseline configuration expects an empty interleaving split")
    return TabularSplits(x[train], y[train], x[validation], y[validation], x[calibration], y[calibration], x[test], y[test])


def download_and_cache_qrt57(upstream_root: str | Path, data_root: str | Path, cache_root: str | Path, seed: int = 0) -> list[Path]:
    """Use upstream download/load functions and cache exact QRT-compatible splits.

    The upstream repository is placed temporarily on `sys.path`; its own
    downloader remains the authoritative dataset source and licensing path.
    """
    root, data, cache = Path(upstream_root), Path(data_root), Path(cache_root)
    if not (root / "uq").is_dir():
        raise FileNotFoundError(f"Missing QRT upstream source at {root}")
    sys.path.insert(0, str(root))
    try:
        from uq.datamodules.openml.download_openml import download_openml_suite, load_dataset as load_openml
        from uq.datamodules.uci.download_uci import download_all_uci, load_dataset as load_uci

        download_all_uci(data)
        for suite in (297, 299, 269):
            download_openml_suite(suite, data)
        saved: list[Path] = []
        for spec in qrt57_manifest():
            loader = load_uci if spec.group == "uci" else load_openml
            source = data / ("uci" if spec.group == "uci" else "openml") / ("" if spec.group == "uci" else spec.group.split("_")[1]) / spec.name
            x, y = loader(source)
            splits = qrt_split(np.asarray(x), np.asarray(y), seed)
            saved.append(save_arrays(cache / f"{spec.name}.npz", **splits.__dict__))
        return saved
    finally:
        sys.path.remove(str(root))
