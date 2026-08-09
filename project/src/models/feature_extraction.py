"""Feature extraction for the NRC-Cal pilot.

Runs a trained pilot model's forward pass over each dataset split and saves
the penultimate-layer activations — the exact tensor NRC geometry (notebook
05, pending formula confirmation) will operate on. No NRC-specific math
lives here; this module's only job is "trained model + data in -> features
out", kept deliberately separate so 05's formula gate doesn't block this
notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from src.models.pilot_mixture_model import PilotMixtureLitModule, load_pilot_checkpoint

logger = logging.getLogger("nrc_cal.models.feature_extraction")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

SPLIT_NAMES = ("train", "val", "calib", "test")


def extract_features_for_array(
    module: PilotMixtureLitModule, x: np.ndarray, batch_size: int = 512, device: str = "cpu",
) -> np.ndarray:
    """Forward-pass `x` through `module` in eval mode, batched, no gradient.

    Parameters
    ----------
    module:
        A model already in eval mode (as returned by :func:`load_pilot_checkpoint`).
    x:
        Input features, shape [N, D].
    batch_size:
        Kept modest by default; feature extraction is cheap regardless.
    device:
        Must match the device `module.model` was moved to.

    Returns
    -------
    np.ndarray
        Penultimate-layer activations, shape [N, hidden_width].
    """
    module.model.eval()
    x_t = torch.from_numpy(x).to(torch.float32).to(device)
    feats = []
    with torch.no_grad():
        for start in range(0, len(x_t), batch_size):
            batch = x_t[start : start + batch_size]
            feats.append(module.penultimate_features(batch).cpu().numpy())
    return np.concatenate(feats, axis=0)


def extract_features_for_all_datasets(
    checkpoint_dir: Path,
    all_splits: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    device: str = "cpu",
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Load each dataset's checkpoint and extract features for all its splits.

    Parameters
    ----------
    checkpoint_dir:
        Directory containing ``<dataset_name>.pt`` files (notebook 03's
        Step 5 output, e.g. ``checkpoints/mixture_1/``).
    all_splits:
        Notebook 02's prepared splits: ``{dataset: {split: {"x":..., "y":...}}}``.
    device:
        Passed through to the loaded model and the forward pass.

    Returns
    -------
    Dict[str, Dict[str, Dict[str, np.ndarray]]]
        ``{dataset: {split: {"features": ..., "y": ...}}}`` — `y` is carried
        through unchanged (needed downstream for PCE, not just features).

    Raises
    ------
    FileNotFoundError
        If a dataset present in `all_splits` has no matching checkpoint file.
    """
    results: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for name, splits in all_splits.items():
        ckpt_path = Path(checkpoint_dir) / f"{name}.pt"
        module = load_pilot_checkpoint(ckpt_path, device=device)
        results[name] = {}
        for split_name in SPLIT_NAMES:
            if split_name not in splits:
                continue
            x = splits[split_name]["x"]
            y = splits[split_name]["y"]
            feats = extract_features_for_array(module, x, device=device)
            results[name][split_name] = {"features": feats, "y": y}
        logger.info(
            "%-10s: extracted features for %s (hidden width=%d)",
            name, list(results[name].keys()),
            results[name]["train"]["features"].shape[1] if "train" in results[name] else -1,
        )
    return results
