"""Unit tests for src.datasets.uci_pilot.

The real UCI archive (archive.ics.uci.edu) is not reachable from this test
sandbox's network allowlist, so the actual download step is not exercised
end-to-end here. Everything downstream of "raw (x, y) arrays already in
memory" — which is where the real reproducibility risk lives (subsampling,
the calibration-cap-and-redistribute arithmetic, scaling) — is fully tested
with synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import uci_pilot  # noqa: E402


def _synthetic_xy(n: int, d: int = 5, seed: int = 0):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, d).astype("float32")
    y = (x[:, :1] * 2 + rng.randn(n, 1).astype("float32") * 0.1)
    return x, y


# --------------------------------------------------------------------------- #
# compute_split_sizes -- the trickiest logic, most important to get right
# --------------------------------------------------------------------------- #


def test_compute_split_sizes_sums_to_total_small():
    """Below the calibration cap: plain ratio split, sums exactly to total."""
    sizes = uci_pilot.compute_split_sizes(total_len=1000)
    assert sum(sizes) == 1000
    assert sizes == [650, 0, 100, 150, 100]  # 15% of 1000 = 150 < 2048, no capping


def test_compute_split_sizes_caps_calibration_and_redistributes():
    """Above the calibration cap (hand-verified against the real formula, see
    module docstring derivation): calib is capped at 2048 and the excess is
    split across train/val/test (not "inter", which has ratio 0)."""
    sizes = uci_pilot.compute_split_sizes(total_len=31328)  # ~ Protein-sized
    train, inter, val, calib, test = sizes
    assert sum(sizes) == 31328
    assert calib == 2048
    assert inter == 0
    # Hand-derived expected values (see uci_pilot.py docstring / design notes):
    assert train == 21246
    assert val == 4016
    assert test == 4018


def test_compute_split_sizes_sums_to_total_across_many_sizes():
    """Property test: for a range of total sizes, the split must always sum
    exactly to total_len and calib must never exceed the cap."""
    for total_len in [50, 500, 2048, 5000, 13657, 31328, 53164, 100000]:
        sizes = uci_pilot.compute_split_sizes(total_len=total_len)
        assert sum(sizes) == total_len, f"failed for total_len={total_len}"
        assert sizes[uci_pilot.CALIB_INDEX] <= uci_pilot.CALIB_CAP
        assert all(s >= 0 for s in sizes), f"negative split size for total_len={total_len}"


def test_compute_split_sizes_rejects_bad_ratio():
    with pytest.raises(ValueError):
        uci_pilot.compute_split_sizes(total_len=1000, split_ratio=(0.5, 0.5, 0.5, 0.0, 0.0))


# --------------------------------------------------------------------------- #
# subsample_to_max_size
# --------------------------------------------------------------------------- #


def test_subsample_is_noop_below_max_size():
    x, y = _synthetic_xy(1000)
    x2, y2 = uci_pilot.subsample_to_max_size(x, y, train_ratio=0.65, seed=0, max_size=50_000)
    assert x2.shape[0] == 1000  # 1000 << 50000/0.65, no-op
    np.testing.assert_array_equal(x, x2)


def test_subsample_reduces_above_max_size():
    x, y = _synthetic_xy(200_000)
    x2, y2 = uci_pilot.subsample_to_max_size(x, y, train_ratio=0.65, seed=0, max_size=50_000)
    expected_keep = int(np.ceil(50_000 / 0.65))
    assert x2.shape[0] == expected_keep
    assert x2.shape[0] < x.shape[0]
    assert y2.shape[0] == x2.shape[0]


# --------------------------------------------------------------------------- #
# StandardScaler
# --------------------------------------------------------------------------- #


def test_standard_scaler_zero_mean_unit_var_on_fit_data():
    x, _ = _synthetic_xy(5000, d=4)
    scaler = uci_pilot.StandardScaler().fit(x)
    transformed = scaler.transform(x)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-5)


def test_standard_scaler_handles_constant_column_without_nan():
    x = np.ones((100, 3), dtype="float32")
    x[:, 1] = np.arange(100)  # only column 1 varies
    scaler = uci_pilot.StandardScaler().fit(x)
    transformed = scaler.transform(x)
    assert not np.isnan(transformed).any()
    np.testing.assert_allclose(transformed[:, 0], 0.0)  # constant column -> exactly 0, not NaN


def test_standard_scaler_raises_if_transform_before_fit():
    scaler = uci_pilot.StandardScaler()
    with pytest.raises(RuntimeError):
        scaler.transform(np.zeros((10, 2)))


# --------------------------------------------------------------------------- #
# split_and_scale (end-to-end on synthetic data)
# --------------------------------------------------------------------------- #


def test_split_and_scale_produces_expected_splits_and_sizes():
    x, y = _synthetic_xy(1000)
    result = uci_pilot.split_and_scale(x, y, seed=0)
    assert set(result.keys()) == {"train", "val", "calib", "test"}  # "inter" dropped

    total = sum(len(result[s]["x"]) for s in result)
    assert total == 1000

    for split in result.values():
        assert split["x"].shape[1] == x.shape[1]
        assert split["y"].shape[1] == y.shape[1]


def test_split_and_scale_train_split_is_standardized():
    x, y = _synthetic_xy(5000)
    result = uci_pilot.split_and_scale(x, y, seed=0)
    np.testing.assert_allclose(result["train"]["x"].mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(result["train"]["x"].std(axis=0), 1.0, atol=1e-5)


def test_split_and_scale_is_reproducible_given_same_seed():
    x, y = _synthetic_xy(1000)
    r1 = uci_pilot.split_and_scale(x, y, seed=42)
    r2 = uci_pilot.split_and_scale(x, y, seed=42)
    np.testing.assert_array_equal(r1["train"]["x"], r2["train"]["x"])


def test_split_and_scale_differs_across_seeds():
    x, y = _synthetic_xy(1000)
    r1 = uci_pilot.split_and_scale(x, y, seed=1)
    r2 = uci_pilot.split_and_scale(x, y, seed=2)
    assert not np.array_equal(r1["train"]["x"], r2["train"]["x"])


# --------------------------------------------------------------------------- #
# add_external_repo_to_path / download_single_uci_dataset
# --------------------------------------------------------------------------- #


def test_add_external_repo_to_path_raises_when_not_cloned(tmp_path):
    with pytest.raises(FileNotFoundError):
        uci_pilot.add_external_repo_to_path(tmp_path)


def test_add_external_repo_to_path_succeeds_when_present(tmp_path):
    repo = tmp_path / "external" / "quantile-recalibration-training"
    repo.mkdir(parents=True)
    result = uci_pilot.add_external_repo_to_path(tmp_path)
    assert result == repo
    assert str(repo) in sys.path
    sys.path.remove(str(repo))  # don't leak into other tests


def test_download_single_uci_dataset_rejects_unknown_name_without_network(tmp_path, monkeypatch):
    """Exercises the name-validation path without touching the network: we
    monkeypatch the external repo onto sys.path with a stub `urls` dict."""
    repo = tmp_path / "external" / "quantile-recalibration-training"
    (repo / "uq" / "datamodules" / "uci").mkdir(parents=True)
    (repo / "uq" / "__init__.py").touch()
    (repo / "uq" / "datamodules" / "__init__.py").touch()
    (repo / "uq" / "datamodules" / "uci" / "__init__.py").touch()
    (repo / "uq" / "datamodules" / "uci" / "download_uci.py").write_text(
        "urls = {'CPU': 'http://example.invalid'}\n"
        "def download_uci(name, url, data_path):\n"
        "    raise AssertionError('should not be called for an unknown name')\n"
    )
    with pytest.raises(ValueError, match="not a known UCI dataset"):
        uci_pilot.download_single_uci_dataset("NotARealDataset", tmp_path, tmp_path)
    sys.path.remove(str(repo))
