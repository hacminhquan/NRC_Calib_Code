"""Dataset preparation for the NRC-Cal pilot (`uci` group).

Faithfully reproduces the splitting/scaling logic found in
``external/quantile-recalibration-training/uq/datamodules/base_datamodule.py``
(read directly, not guessed) so that PCE numbers computed later are
comparable to the paper's. The raw per-dataset download/parsing itself is
**not** re-implemented here — it is imported directly from the cloned
upstream repo (``uq.datamodules.uci.download_uci``), since that logic is
bespoke per dataset (12 different file formats/parsers) and re-typing it
by hand would be exactly the kind of silent re-derivation this project's
own rules say to avoid.

Two things this module does NOT get from upstream, because upstream
doesn't expose them as reusable functions on their own — they are ported
here as direct, faithful translations of the exact arithmetic in
``BaseDataModule.subsample`` and ``BaseDataModule.load_datasets``:

- :func:`subsample_to_max_size`
- :func:`compute_split_sizes` (the "cap calibration at 2048, redistribute
  the excess to the other non-empty splits" rule)
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nrc_cal.datasets.uci_pilot")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# Verified against uq/configs/dataset_groups.py (notebook 01, Step 5).
UCI_PILOT_DATASETS: Tuple[str, ...] = (
    "CPU", "Yacht", "MPG", "Energy", "Crime", "Fish",
    "Concrete", "Airfoil", "Kin8nm", "Power", "Naval", "Protein",
)

# Verified against uq/configs/dataset_groups.py / uq/configs/general.py.
DEFAULT_SPLIT_RATIO: Tuple[float, float, float, float, float] = (0.65, 0.0, 0.1, 0.15, 0.1)
DEFAULT_BATCH_SIZE = 512
CALIB_CAP = 2048
CALIB_INDEX = 3  # position of "calib" within [train, inter, val, calib, test]


# --------------------------------------------------------------------------- #
# Importing the real upstream download logic (not re-derived)
# --------------------------------------------------------------------------- #


def add_external_repo_to_path(project_root: Path) -> Path:
    """Put the cloned QRT repo on ``sys.path`` so its real modules import cleanly.

    Parameters
    ----------
    project_root:
        This project's root (containing ``external/``).

    Returns
    -------
    Path
        The path that was added.

    Raises
    ------
    FileNotFoundError
        If the external repo has not been cloned yet (run notebook 01 first).
    """
    repo_path = project_root / "external" / "quantile-recalibration-training"
    if not repo_path.exists():
        raise FileNotFoundError(
            f"{repo_path} not found -- run 01_download_repositories.ipynb first."
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    return repo_path


def download_single_uci_dataset(name: str, data_dir: Path, project_root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Download (or load from cache) one named UCI dataset via the real upstream code.

    Thin wrapper around ``uq.datamodules.uci.download_uci.download_uci`` — the
    actual per-format parsing (12 different file types) is upstream's, not
    reimplemented here.

    Parameters
    ----------
    name:
        One of :data:`UCI_PILOT_DATASETS` (or any key in upstream's ``urls`` dict).
    data_dir:
        Cache directory upstream's code reads/writes ``x.npy`` / ``y.npy`` under
        (mirrors their own ``data_dir / 'uci' / name`` convention).
    project_root:
        Used to locate the cloned external repo.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Raw (unsplit, unscaled) ``x`` (float32, [N, D]) and ``y`` (float32, [N, 1]).
    """
    add_external_repo_to_path(project_root)
    from uq.datamodules.uci.download_uci import download_uci, urls  # type: ignore[import-not-found]

    if name not in urls:
        raise ValueError(f"'{name}' is not a known UCI dataset name. Known: {sorted(urls)}")

    logger.info("Fetching '%s' (cached under %s if already downloaded)...", name, data_dir / "uci" / name)
    x, y = download_uci(name, urls[name], data_dir)
    return x, y


# --------------------------------------------------------------------------- #
# Faithful ports of BaseDataModule.subsample / load_datasets arithmetic
# --------------------------------------------------------------------------- #


def subsample_to_max_size(
    x: np.ndarray, y: np.ndarray, train_ratio: float, seed: int, max_size: int = 50_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Port of ``BaseDataModule.subsample``: cap total dataset size before splitting.

    Only matters for datasets larger than ``max_size / train_ratio`` rows; none
    of the 12 `uci` pilot datasets are (largest is Protein at ~31k rows), so
    this is a no-op for the pilot but kept faithful for when this module is
    reused on the larger OpenML dataset groups.
    """
    n = x.shape[0]
    rng = np.random.RandomState(seed + 1)  # upstream uses seed+1 here specifically
    keep = min(n, math.ceil(max_size / train_ratio))
    if keep == n:
        return x, y
    idx = rng.choice(n, keep, replace=False)
    return x[idx], y[idx]


def compute_split_sizes(
    total_len: int,
    split_ratio: Tuple[float, ...] = DEFAULT_SPLIT_RATIO,
    calib_cap: int = CALIB_CAP,
    calib_index: int = CALIB_INDEX,
) -> List[int]:
    """Port of the split-size arithmetic inside ``BaseDataModule.load_datasets``.

    Converts ``split_ratio`` into integer sample counts for
    ``[train, inter, val, calib, test]``, capping the calibration split at
    ``calib_cap`` samples and redistributing the excess proportionally across
    the other *non-empty, non-calibration* splits — exactly as upstream does,
    not an approximation of it.

    Parameters
    ----------
    total_len:
        Total number of samples in the (post-subsampling) dataset.
    split_ratio:
        5-tuple of ratios (train, inter, val, calib, test); must sum to 1.0.
    calib_cap:
        Maximum number of samples allowed in the calibration split.
    calib_index:
        Index of the calibration split within ``split_ratio``.

    Returns
    -------
    List[int]
        Integer sample counts, same length as ``split_ratio``, summing
        exactly to ``total_len``.
    """
    if not math.isclose(sum(split_ratio), 1.0, abs_tol=1e-6):
        raise ValueError(f"split_ratio must sum to 1.0, got {sum(split_ratio)} from {split_ratio}")

    splits_size = np.array(split_ratio, dtype=float) * total_len

    to_remove_from_calib = max(0.0, splits_size[calib_index] - calib_cap)
    splits_size[calib_index] -= to_remove_from_calib

    mask = (splits_size != 0) & (np.arange(len(splits_size)) != calib_index)
    if to_remove_from_calib > 0 and mask.sum() == 0:
        raise ValueError(
            "Calibration split exceeds the cap but there are no other non-empty "
            "splits to redistribute the excess into -- check split_ratio."
        )
    if mask.sum() > 0:
        splits_size[mask] += to_remove_from_calib / mask.sum()

    splits_size = splits_size.astype(int)
    splits_size[-1] = total_len - splits_size[:-1].sum()  # last split absorbs rounding remainder
    return splits_size.tolist()


class StandardScaler:
    """Faithful, framework-free port of upstream's train-only StandardScaler use.

    Deliberately not importing sklearn's here to keep this module's only real
    dependency on upstream code confined to the download step above — this
    is a direct, minimal port of the same fit-on-train/transform-everywhere
    pattern ``BaseDataModule.load_datasets`` uses (via sklearn.preprocessing
    there; behaviorally identical here).
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        std[std == 0] = 1.0  # avoid division by zero on constant columns
        self.scale_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler.transform called before fit().")
        return (x - self.mean_) / self.scale_


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def split_and_scale(
    x: np.ndarray,
    y: np.ndarray,
    split_ratio: Tuple[float, ...] = DEFAULT_SPLIT_RATIO,
    seed: int = 0,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Subsample, split, and scale one dataset's raw ``(x, y)`` arrays.

    Mirrors ``BaseDataModule.load_datasets``: subsample -> compute split
    sizes -> random split with a seeded generator -> fit scalers on the
    train split only -> apply to every split.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        ``{"train": {"x":..., "y":...}, "val": {...}, "calib": {...}, "test": {...}}``.
        The (always-empty, ratio=0) "inter" split is dropped from the output
        since nothing downstream in this project uses it.
    """
    train_ratio = split_ratio[0]
    x, y = subsample_to_max_size(x, y, train_ratio=train_ratio, seed=seed)

    total_len = x.shape[0]
    sizes = compute_split_sizes(total_len, split_ratio=split_ratio)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(total_len)

    names = ["train", "inter", "val", "calib", "test"]
    splits: Dict[str, Dict[str, np.ndarray]] = {}
    start = 0
    for name, size in zip(names, sizes):
        idx = perm[start : start + size]
        start += size
        if name == "inter":
            continue  # always empty under DEFAULT_SPLIT_RATIO; upstream keeps it, we drop it
        splits[name] = {"x": x[idx], "y": y[idx]}

    scaler_x = StandardScaler().fit(splits["train"]["x"])
    scaler_y = StandardScaler().fit(splits["train"]["y"])
    for name in splits:
        splits[name]["x"] = scaler_x.transform(splits[name]["x"])
        splits[name]["y"] = scaler_y.transform(splits[name]["y"])

    return splits


def prepare_uci_pilot_group(
    project_root: Path,
    data_dir: Path,
    dataset_names: Tuple[str, ...] = UCI_PILOT_DATASETS,
    split_ratio: Tuple[float, ...] = DEFAULT_SPLIT_RATIO,
    seed: int = 0,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Download, split, and scale every dataset in the `uci` pilot group.

    Parameters
    ----------
    project_root:
        This project's root (for locating the cloned external repo).
    data_dir:
        Cache directory for raw downloads (upstream's convention: ``uci/<name>/``
        subfolders under this).
    dataset_names:
        Which datasets to prepare; defaults to the full 12-dataset `uci` group.
    split_ratio, seed:
        Passed through to :func:`split_and_scale`.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, np.ndarray]]]
        ``{dataset_name: {"train": {"x":..., "y":...}, "val": {...}, ...}}``.
    """
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for name in dataset_names:
        try:
            x, y = download_single_uci_dataset(name, data_dir, project_root)
        except Exception as exc:
            logger.error("Failed to download/prepare '%s': %s", name, exc)
            raise
        results[name] = split_and_scale(x, y, split_ratio=split_ratio, seed=seed)
        n_train = len(results[name]["train"]["x"])
        n_calib = len(results[name]["calib"]["x"])
        logger.info(
            "%-10s: total=%d  train=%d val=%d calib=%d test=%d",
            name, x.shape[0], n_train, len(results[name]["val"]["x"]), n_calib, len(results[name]["test"]["x"]),
        )
    return results
