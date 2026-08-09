"""Unit tests for checkpoint loading (pilot_mixture_model.load_pilot_checkpoint)
and src.models.feature_extraction.

Uses the real upstream MixturePrediction class (see test_pilot_mixture_model.py
for why) and real, small, fast training runs -- not mocks -- so the
save -> load -> extract round trip is genuinely exercised end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models import pilot_mixture_model as pmm  # noqa: E402
from src.models import feature_extraction as fe  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_REPO = PROJECT_ROOT / "external" / "quantile-recalibration-training"
requires_upstream = pytest.mark.skipif(
    not EXTERNAL_REPO.exists(),
    reason="external/quantile-recalibration-training not cloned (run notebook 01 first)",
)


@pytest.fixture(scope="module", autouse=True)
def _wire_real_mixture_prediction():
    if not EXTERNAL_REPO.exists():
        pytest.skip("external repo not present")
    cls = pmm.import_mixture_prediction(PROJECT_ROOT)
    pmm.set_mixture_prediction_cls(cls)
    yield


def _toy_splits(n_train=200, n_val=60, n_calib=60, n_test=60, d=4, seed=0):
    rng = np.random.RandomState(seed)

    def make(n):
        x = rng.randn(n, d).astype("float32")
        y = (x.sum(axis=1, keepdims=True) + rng.randn(n, 1).astype("float32") * 0.05).astype("float32")
        return {"x": x, "y": y}

    return {"train": make(n_train), "val": make(n_val), "calib": make(n_calib), "test": make(n_test)}


@pytest.fixture()
def trained_checkpoint(tmp_path):
    """Trains one small real model and saves it exactly as notebook 03 does."""
    splits = _toy_splits()
    result = pmm.train_pilot_model(splits, mixture_size=1, seed=0, patience=5, max_epochs=60, verbose=False)
    module = result["module"]
    ckpt_path = tmp_path / "toy_dataset.pt"
    torch.save(
        {
            "state_dict": module.model.state_dict(),
            "input_size": module.input_size,
            "mixture_size": module.mixture_size,
            "best_val_nll": result["best_val_nll"],
            "best_epoch": result["best_epoch"],
            "seed": 0,
        },
        ckpt_path,
    )
    return ckpt_path, module, splits


# --------------------------------------------------------------------------- #
# load_pilot_checkpoint
# --------------------------------------------------------------------------- #


def test_load_pilot_checkpoint_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        pmm.load_pilot_checkpoint(tmp_path / "does_not_exist.pt")


def test_load_pilot_checkpoint_raises_on_malformed_checkpoint(tmp_path):
    bad_ckpt = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, bad_ckpt)  # missing input_size/mixture_size/seed
    with pytest.raises(KeyError):
        pmm.load_pilot_checkpoint(bad_ckpt)


@requires_upstream
def test_load_pilot_checkpoint_reproduces_original_predictions(trained_checkpoint):
    """The real correctness check: loaded model's outputs must exactly match
    the original (pre-save) trained model's outputs on the same input."""
    ckpt_path, original_module, splits = trained_checkpoint
    loaded_module = pmm.load_pilot_checkpoint(ckpt_path)

    x = torch.from_numpy(splits["test"]["x"]).to(torch.float32)
    original_module.model.eval()
    with torch.no_grad():
        orig_means, orig_stds, _ = original_module.model(x)
        loaded_means, loaded_stds, _ = loaded_module.model(x)

    torch.testing.assert_close(orig_means, loaded_means)
    torch.testing.assert_close(orig_stds, loaded_stds)


@requires_upstream
def test_load_pilot_checkpoint_sets_eval_mode(trained_checkpoint):
    ckpt_path, _, _ = trained_checkpoint
    loaded_module = pmm.load_pilot_checkpoint(ckpt_path)
    assert loaded_module.model.training is False


# --------------------------------------------------------------------------- #
# extract_features_for_array
# --------------------------------------------------------------------------- #


@requires_upstream
def test_extract_features_for_array_shape(trained_checkpoint):
    ckpt_path, _, splits = trained_checkpoint
    module = pmm.load_pilot_checkpoint(ckpt_path)
    feats = fe.extract_features_for_array(module, splits["test"]["x"])
    assert feats.shape == (len(splits["test"]["x"]), pmm.DEFAULT_HIDDEN_SIZES[-1])


@requires_upstream
def test_extract_features_deterministic_in_eval_mode(trained_checkpoint):
    """Eval mode disables dropout -- running twice must give identical features."""
    ckpt_path, _, splits = trained_checkpoint
    module = pmm.load_pilot_checkpoint(ckpt_path)
    f1 = fe.extract_features_for_array(module, splits["test"]["x"])
    f2 = fe.extract_features_for_array(module, splits["test"]["x"])
    np.testing.assert_array_equal(f1, f2)


@requires_upstream
def test_extract_features_batching_does_not_change_result(trained_checkpoint):
    """Splitting into small batches must give the same features as one big batch
    -- guards against any hidden cross-sample leakage in the forward pass."""
    ckpt_path, _, splits = trained_checkpoint
    module = pmm.load_pilot_checkpoint(ckpt_path)
    x = splits["test"]["x"]
    f_one_batch = fe.extract_features_for_array(module, x, batch_size=10_000)
    f_small_batches = fe.extract_features_for_array(module, x, batch_size=7)
    np.testing.assert_allclose(f_one_batch, f_small_batches, atol=1e-5)


# --------------------------------------------------------------------------- #
# extract_features_for_all_datasets (orchestration)
# --------------------------------------------------------------------------- #


@requires_upstream
def test_extract_features_for_all_datasets_end_to_end(tmp_path):
    all_splits = {"toy_a": _toy_splits(seed=1), "toy_b": _toy_splits(seed=2, d=3)}
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()

    for name, splits in all_splits.items():
        result = pmm.train_pilot_model(splits, mixture_size=1, seed=0, patience=3, max_epochs=30, verbose=False)
        module = result["module"]
        torch.save(
            {"state_dict": module.model.state_dict(), "input_size": module.input_size,
             "mixture_size": module.mixture_size, "seed": 0, "best_val_nll": result["best_val_nll"],
             "best_epoch": result["best_epoch"]},
            ckpt_dir / f"{name}.pt",
        )

    extracted = fe.extract_features_for_all_datasets(ckpt_dir, all_splits)

    assert set(extracted.keys()) == {"toy_a", "toy_b"}
    for name, splits in all_splits.items():
        for split_name in ("train", "val", "calib", "test"):
            n_expected = len(splits[split_name]["x"])
            assert extracted[name][split_name]["features"].shape[0] == n_expected
            np.testing.assert_array_equal(extracted[name][split_name]["y"], splits[split_name]["y"])


def test_extract_features_for_all_datasets_raises_on_missing_checkpoint(tmp_path):
    all_splits = {"missing_dataset": _toy_splits()}
    (tmp_path / "empty_ckpt_dir").mkdir()
    with pytest.raises(FileNotFoundError):
        fe.extract_features_for_all_datasets(tmp_path / "empty_ckpt_dir", all_splits)
