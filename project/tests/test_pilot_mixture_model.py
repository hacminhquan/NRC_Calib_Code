"""Unit tests for src.models.pilot_mixture_model.

Uses the *real* `MixturePrediction` class imported from the cloned upstream
repo (staged under external/ for this test run) rather than a hand-written
stand-in, so these tests exercise the actual architecture, not a guess at
its behavior. Training is exercised on small synthetic regression problems
(no network dependency at all -- unlike notebook 02, training itself never
needs to reach the internet).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import pilot_mixture_model as pmm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO = PROJECT_ROOT / "external" / "quantile-recalibration-training"
requires_upstream = pytest.mark.skipif(
    not EXTERNAL_REPO.exists(),
    reason="external/quantile-recalibration-training not cloned (run notebook 01 first)",
)


@pytest.fixture(scope="module", autouse=True)
def _wire_real_mixture_prediction():
    """Import the real MixturePrediction once for this test module."""
    if not EXTERNAL_REPO.exists():
        pytest.skip("external repo not present")
    cls = pmm.import_mixture_prediction(PROJECT_ROOT)
    pmm.set_mixture_prediction_cls(cls)
    yield


def _toy_splits(n_train=400, n_val=100, n_calib=100, n_test=100, d=3, seed=0, noise=0.05):
    """A simple, learnable synthetic regression problem: y = sum(x) + small noise."""
    rng = np.random.RandomState(seed)

    def make(n):
        x = rng.randn(n, d).astype("float32")
        y = (x.sum(axis=1, keepdims=True) + rng.randn(n, 1).astype("float32") * noise).astype("float32")
        return {"x": x, "y": y}

    return {"train": make(n_train), "val": make(n_val), "calib": make(n_calib), "test": make(n_test)}


# --------------------------------------------------------------------------- #
# import_mixture_prediction
# --------------------------------------------------------------------------- #


def test_import_mixture_prediction_raises_when_repo_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pmm.import_mixture_prediction(tmp_path)


@requires_upstream
def test_import_mixture_prediction_returns_real_class():
    cls = pmm.import_mixture_prediction(PROJECT_ROOT)
    assert cls.__name__ == "MixturePrediction"
    assert cls.__module__ == "uq.models.general.mlp"


# --------------------------------------------------------------------------- #
# PilotMixtureLitModule -- architecture shape checks
# --------------------------------------------------------------------------- #


@requires_upstream
@pytest.mark.parametrize("mixture_size", [1, 3])
def test_module_builds_with_verified_default_hidden_sizes(mixture_size):
    module = pmm.PilotMixtureLitModule(input_size=5, mixture_size=mixture_size, seed=0)
    hidden_sizes = [layer.out_features for layer in module.model.body.hidden_layers]
    assert hidden_sizes == list(pmm.DEFAULT_HIDDEN_SIZES)


@requires_upstream
def test_module_output_layer_width_matches_mixture_math():
    """For event_size=1, mixture_size=K: output = means(K) + rhos(K) + mix_logits(K) = 3K."""
    K = 4
    module = pmm.PilotMixtureLitModule(input_size=5, mixture_size=K, seed=0)
    assert module.model.body.output_layer.out_features == 3 * K


@requires_upstream
def test_nll_loss_is_finite_and_differentiable():
    module = pmm.PilotMixtureLitModule(input_size=3, mixture_size=3, seed=0)
    x = torch.randn(16, 3)
    y = torch.randn(16, 1)
    loss = module.nll_loss(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in module.model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)


@requires_upstream
def test_penultimate_features_shape_matches_last_hidden_width():
    module = pmm.PilotMixtureLitModule(input_size=5, mixture_size=3, seed=0)
    x = torch.randn(10, 5)
    feats = module.penultimate_features(x)
    assert feats.shape == (10, pmm.DEFAULT_HIDDEN_SIZES[-1])


@requires_upstream
def test_same_seed_gives_identical_initial_weights():
    m1 = pmm.PilotMixtureLitModule(input_size=4, mixture_size=3, seed=7)
    m2 = pmm.PilotMixtureLitModule(input_size=4, mixture_size=3, seed=7)
    for p1, p2 in zip(m1.model.parameters(), m2.model.parameters()):
        torch.testing.assert_close(p1, p2)


# --------------------------------------------------------------------------- #
# make_dataloaders
# --------------------------------------------------------------------------- #


def test_make_dataloaders_respects_split_sizes_and_batch_cap():
    splits = _toy_splits(n_train=400, n_val=5)  # val smaller than default batch_size=512
    loaders = pmm.make_dataloaders(splits, batch_size=512)
    assert len(loaders["train"].dataset) == 400
    assert len(loaders["val"].dataset) == 5
    # batch size is capped to split size when the split is smaller than batch_size
    assert loaders["val"].batch_size == 5


def test_make_dataloaders_only_train_shuffles():
    splits = _toy_splits()
    loaders = pmm.make_dataloaders(splits)
    assert loaders["train"].sampler.__class__.__name__ != "SequentialSampler"
    assert loaders["val"].sampler.__class__.__name__ == "SequentialSampler"


# --------------------------------------------------------------------------- #
# train_pilot_model -- the real, meaningful sanity check
# --------------------------------------------------------------------------- #


@requires_upstream
def test_train_pilot_model_reduces_validation_nll_on_learnable_toy_problem():
    """The single most important test in this file: real gradient descent,
    on a real (if tiny) learnable problem, using the real architecture,
    must actually reduce validation NLL — not just run without crashing."""
    splits = _toy_splits(n_train=500, n_val=150, d=3, noise=0.05)
    result = pmm.train_pilot_model(
        splits, mixture_size=1, seed=0, patience=15, max_epochs=300, verbose=False,
    )
    history = result["history"]
    assert len(history["val_nll"]) >= 2
    # Best val NLL must be substantially better than the first epoch's.
    assert result["best_val_nll"] < history["val_nll"][0] - 0.5
    assert result["best_epoch"] >= 0


@requires_upstream
def test_train_pilot_model_early_stopping_actually_stops():
    splits = _toy_splits(n_train=200, n_val=60)
    result = pmm.train_pilot_model(
        splits, mixture_size=1, seed=0, patience=3, max_epochs=1000, verbose=False,
    )
    # With patience=3 on a small, quickly-converging toy problem, training
    # must stop well short of the 1000-epoch cap.
    assert len(result["history"]["val_nll"]) < 1000


@requires_upstream
def test_train_pilot_model_restores_best_weights_not_last():
    splits = _toy_splits(n_train=300, n_val=80)
    result = pmm.train_pilot_model(
        splits, mixture_size=1, seed=0, patience=5, max_epochs=200, verbose=False,
    )
    module = result["module"]
    # Recompute val NLL from the restored module's weights and confirm it
    # matches the recorded best (not the final pre-restore epoch's) value.
    loaders = pmm.make_dataloaders(splits)
    module.model.eval()
    with torch.no_grad():
        val_losses = [module.nll_loss(x, y).item() for x, y in loaders["val"]]
    recomputed = float(np.mean(val_losses))
    assert abs(recomputed - result["best_val_nll"]) < 1e-3
