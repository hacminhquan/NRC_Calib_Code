"""Artifact-driven, frozen-model NRC-Cal experiment execution."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.nrc_cal import select_ridge
from geometry.nrc import compute_nrc
from metrics.evaluation import evaluate
from models.predictions import GaussianMixturePrediction
from utils.io import save_frame, save_json


def load_prediction_cache(path: str | Path) -> tuple[GaussianMixturePrediction, np.ndarray]:
    """Load a validated prediction cache produced by the feature notebook.

    Required keys are `weights`, `means`, `covariances`, and `targets`; this
    intentionally rejects ambiguous model-output encodings instead of guessing
    whether a tensor represents a variance, standard deviation, or log scale.
    """
    cache = np.load(Path(path))
    required = {"weights", "means", "covariances", "targets"}
    missing = required.difference(cache.files)
    if missing:
        raise KeyError(f"Prediction cache missing required arrays: {sorted(missing)}")
    return GaussianMixturePrediction(cache["weights"], cache["means"], cache["covariances"]), np.asarray(cache["targets"])


def run_frozen_nrc_cal(dataset: str, family: str, calibration_feature_cache: str | Path, calibration_prediction_cache: str | Path, test_feature_cache: str | Path, test_prediction_cache: str | Path, mean_head_weight: np.ndarray, output_path: str | Path) -> pd.DataFrame:
    """Execute BASE and NRC-Cal from immutable feature/prediction artifacts.

    This runner never trains or changes a checkpoint. It is model-family
    agnostic because all Gaussian and mixture heads are normalized into the
    `GaussianMixturePrediction` artifact contract.
    """
    calibration_features = np.load(Path(calibration_feature_cache))
    test_features = np.load(Path(test_feature_cache))
    calibration_prediction, calibration_targets = load_prediction_cache(calibration_prediction_cache)
    test_prediction, test_targets = load_prediction_cache(test_prediction_cache)
    calibration_nrc = compute_nrc(calibration_features["features"], calibration_features["targets"], mean_head_weight)
    test_nrc = compute_nrc(test_features["features"], test_features["targets"], mean_head_weight)
    calibrator = select_ridge(calibration_prediction, calibration_targets, calibration_nrc.sample_distance)
    calibrated_test = calibrator.transform(test_prediction, test_nrc.sample_distance)
    rows: list[dict[str, object]] = []
    for method, prediction in (("BASE", test_prediction), ("NRC-Cal", calibrated_test)):
        row: dict[str, object] = {"dataset": dataset, "family": family, "method": method, "nrc_distance": test_nrc.dataset_distance, "nrc1": test_nrc.nrc1, "nrc2": test_nrc.nrc2, "nrc3": test_nrc.nrc3, "nrc_gamma": test_nrc.gamma}
        row.update(evaluate(prediction, test_targets)); rows.append(row)
    frame = pd.DataFrame(rows)
    save_frame(frame, output_path)
    save_json({"dataset": dataset, "family": family, "nrc_calibrator": asdict(calibrator), "published_nrc_normalization": test_nrc.normalization, "nrc_distance_is_proposed": True}, Path(output_path).with_suffix(".provenance.json"))
    return frame
