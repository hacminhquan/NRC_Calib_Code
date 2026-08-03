"""Atomic, typed artifact persistence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Atomically write JSON and return its normalized path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    """Save a CSV result table with a stable schema and no index."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination


def save_arrays(path: str | Path, **arrays: np.ndarray) -> Path:
    """Persist named arrays in a compressed NPZ feature cache."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return destination
