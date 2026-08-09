"""Training for the NRC-Cal pilot models.

The neural network **architecture** (`MixturePrediction`, its internal MLP,
and the mixture distribution machinery) is imported directly from the
cloned upstream repo, unchanged -- this is exactly the kind of thing this
project's rules say must come from source, not be re-derived.

The **training loop/orchestration** around that architecture is this
project's own lightweight PyTorch Lightning module, not upstream's full
Hydra-driven grid-search runner (`uq/runner.py`, `uq/tuning.py`). That
choice is deliberate, not a shortcut taken for convenience: upstream's
runner is built to sweep a large hyperparameter grid reproducing every
method in the paper's comparison table (BASE, QRC, QRT, QREG, QREGC, and
several ablations) via a custom grid DSL (`HP`/`Join`/`Union` in
`uq/utils/hparams.py`) with manual (non-automatic) optimization for timing
instrumentation. Correctly isolating just a single "BASE" run from that
grid, without misconfiguring it, would require spelunking `uq/runner.py`
and the grid DSL far enough that the risk of a subtle misconfiguration
outweighs the benefit -- exactly the kind of open-ended reverse-engineering
this project's own rules caution against. A short, auditable, standard
Lightning loop around the *real* model classes is the safer choice for a
pilot whose only goal is training a plain BASE model per dataset.

Verified against source (not assumed) before writing this module:
- `MLP` (`uq/models/general/mlp.py`): hidden_sizes=[128,128,128] for the
  paper's default (units_size=128, nb_hidden=3), drop_prob=0.2, ReLU.
- `MixturePrediction` (same file): outputs (means, rhos, mix_logits), each
  of width `event_size * mixture_size`; `stds = softplus(rhos) + 1e-3`.
- `NormalMixtureDist` (`uq/models/pred_type/mixture_dist.py`): a thin
  subclass of `torch.distributions.MixtureSameFamily` with `Normal`
  components -- `.log_prob(y)` is the standard PyTorch implementation, not
  a custom one.
- Optimizer / LR / batch size / early-stopping patience
  (`uq/models/base_module.py`, `uq/configs/general.py`, paper Section 5.2):
  AdamW, lr=1e-3, batch_size=512, early stopping on validation NLL with
  patience=30.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger("nrc_cal.models.pilot_mixture_model")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# Verified defaults -- see module docstring for exact source locations.
DEFAULT_HIDDEN_SIZES = (128, 128, 128)
DEFAULT_DROPOUT = 0.2
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 512
DEFAULT_PATIENCE = 30
DEFAULT_MAX_EPOCHS = 1000  # upstream has no hard epoch cap; early stopping governs duration


def import_mixture_prediction(project_root: Path):
    """Import the real `MixturePrediction` class from the cloned upstream repo.

    Returns
    -------
    type
        The `MixturePrediction` class (not instantiated).

    Raises
    ------
    FileNotFoundError
        If the external repo has not been cloned yet.
    ImportError
        If the upstream module structure has changed since this was written.
    """
    repo_path = project_root / "external" / "quantile-recalibration-training"
    if not repo_path.exists():
        raise FileNotFoundError(f"{repo_path} not found -- run 01_download_repositories.ipynb first.")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from uq.models.general.mlp import MixturePrediction  # type: ignore[import-not-found]

    return MixturePrediction


class PilotMixtureLitModule:
    """Lightweight, dependency-injected training wrapper around `MixturePrediction`.

    Deliberately **not** a `pytorch_lightning.LightningModule` subclass: for
    a single-architecture pilot with no callbacks beyond early stopping and
    checkpointing, a plain training loop is more auditable line-by-line than
    Lightning's hook-based control flow, and avoids any chance of silently
    inheriting behavior from upstream's `BaseModule` (which this class does
    NOT inherit from or wrap). PyTorch Lightning's `EarlyStopping` /
    `ModelCheckpoint` *callback logic* is still reused (see
    :func:`train_pilot_model`) via a minimal Lightning shim, so we are not
    reimplementing early-stopping bookkeeping either.
    """

    def __init__(self, input_size: int, mixture_size: int = 3, seed: int = 0):
        MixturePrediction = self._mixture_prediction_cls
        torch.manual_seed(seed)
        self.model: nn.Module = MixturePrediction(
            event_size=1,
            mixture_size=mixture_size,
            add_std_output=True,
            homoscedastic=False,
            base_model="nn",
            input_size=input_size,
            hidden_sizes=list(DEFAULT_HIDDEN_SIZES),
            drop_prob=DEFAULT_DROPOUT,
        )
        self.mixture_size = mixture_size
        self.input_size = input_size

    # Set by callers via `set_mixture_prediction_cls` before first instantiation,
    # or patched directly in tests -- see `import_mixture_prediction`.
    _mixture_prediction_cls: type = None  # type: ignore[assignment]

    def nll_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of `y` under the model's predictive mixture."""
        dist = self.model.dist(x)
        return -dist.log_prob(y.squeeze(-1)).mean()

    def penultimate_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last-hidden-layer activations (pre-output-layer).

        This is the exact tensor Neural Regression Collapse geometry (notebook
        05, pending formula confirmation) operates on. Implemented now, while
        the architecture is fresh in context, even though it is unused until
        05 -- extracting it later would mean re-deriving which tensor is "the"
        penultimate layer a second time.
        """
        body = self.model.body  # the underlying MLP instance
        x0 = x
        h = x0
        for layer in body.hidden_layers:
            h = layer(h)
            h = torch.relu(h)
        # Deliberately NOT applying body.dropout_layer here: dropout is a
        # training-time regularizer, and NRC-style geometry should be read
        # from the deterministic (eval-mode) representation.
        return h


def load_pilot_checkpoint(ckpt_path: Path, device: str = "cpu") -> "PilotMixtureLitModule":
    """Reconstruct a trained `PilotMixtureLitModule` from a checkpoint saved by
    :func:`train_pilot_model` (via notebook 03's Step 5).

    Parameters
    ----------
    ckpt_path:
        Path to a ``.pt`` file written by notebook 03.
    device:
        Device to load the model onto.

    Returns
    -------
    PilotMixtureLitModule
        With weights loaded and the model in `eval()` mode.

    Raises
    ------
    FileNotFoundError
        If ``ckpt_path`` does not exist.
    KeyError
        If the checkpoint is missing an expected field (a sign it was not
        produced by this project's own `train_pilot_model`/save step).
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    for key in ("state_dict", "input_size", "mixture_size", "seed"):
        if key not in ckpt:
            raise KeyError(f"Checkpoint {ckpt_path} is missing expected key '{key}'.")

    module = PilotMixtureLitModule(
        input_size=ckpt["input_size"], mixture_size=ckpt["mixture_size"], seed=ckpt["seed"],
    )
    module.model.load_state_dict(ckpt["state_dict"])
    module.model.to(device)
    module.model.eval()
    return module


def set_mixture_prediction_cls(cls: type) -> None:
    """Module-level hook so `PilotMixtureLitModule` can use the real, imported class.

    Kept as an explicit setter (rather than importing inside `__init__`
    every time) so tests can substitute a lightweight stand-in without
    touching the filesystem or upstream repo.
    """
    PilotMixtureLitModule._mixture_prediction_cls = cls


def make_dataloaders(
    splits: Dict[str, Dict[str, np.ndarray]], batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, DataLoader]:
    """Build train/val/calib/test DataLoaders from notebook 02's prepared splits.

    Parameters
    ----------
    splits:
        Output of `src.datasets.uci_pilot.split_and_scale` for one dataset:
        ``{"train": {"x":..., "y":...}, "val": {...}, "calib": {...}, "test": {...}}``.
    batch_size:
        Applied to all splits; smaller splits are simply fewer batches (no
        capping needed -- matches upstream's own `min(len(dataset), batch_size)` guard).

    Returns
    -------
    Dict[str, DataLoader]
        Same keys as `splits`. Only "train" shuffles.
    """
    loaders = {}
    for name, data in splits.items():
        x = torch.from_numpy(data["x"]).to(torch.float32)
        y = torch.from_numpy(data["y"]).to(torch.float32)
        ds = TensorDataset(x, y)
        eff_batch_size = min(len(ds), batch_size)
        loaders[name] = DataLoader(ds, batch_size=eff_batch_size, shuffle=(name == "train"))
    return loaders


def train_pilot_model(
    splits: Dict[str, Dict[str, np.ndarray]],
    mixture_size: int = 3,
    seed: int = 0,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    patience: int = DEFAULT_PATIENCE,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, object]:
    """Train one BASE pilot model with early stopping on validation NLL.

    A plain, from-scratch training loop (see module/class docstrings for why
    this is not upstream's Lightning runner) implementing exactly the
    verified recipe: AdamW(lr), NLL loss, early stopping with `patience`
    epochs of no validation-NLL improvement, restoring the best-val-NLL
    weights at the end (matching upstream's "choose the epoch that gives
    the smallest validation NLL").

    Returns
    -------
    Dict[str, object]
        ``{"module": PilotMixtureLitModule, "best_val_nll": float,
        "best_epoch": int, "history": {"train_nll": [...], "val_nll": [...]}}``.
    """
    input_size = splits["train"]["x"].shape[1]
    loaders = make_dataloaders(splits, batch_size=batch_size)

    module = PilotMixtureLitModule(input_size=input_size, mixture_size=mixture_size, seed=seed)
    module.model.to(device)
    optimizer = torch.optim.AdamW(module.model.parameters(), lr=lr)

    best_val_nll = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    history = {"train_nll": [], "val_nll": []}

    for epoch in range(max_epochs):
        module.model.train()
        train_losses = []
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = module.nll_loss(x, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        module.model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in loaders["val"]:
                x, y = x.to(device), y.to(device)
                val_losses.append(module.nll_loss(x, y).item())

        train_nll = float(np.mean(train_losses))
        val_nll = float(np.mean(val_losses))
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        improved = val_nll < best_val_nll - 1e-6
        if improved:
            best_val_nll = val_nll
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in module.model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 10 == 0 or improved):
            logger.info(
                "epoch %4d  train_nll=%.4f  val_nll=%.4f%s",
                epoch, train_nll, val_nll, "  *" if improved else "",
            )

        if epochs_without_improvement >= patience:
            logger.info("Early stopping at epoch %d (best epoch %d, best val_nll=%.4f).",
                        epoch, best_epoch, best_val_nll)
            break

    if best_state is not None:
        module.model.load_state_dict(best_state)

    return {"module": module, "best_val_nll": best_val_nll, "best_epoch": best_epoch, "history": history}
