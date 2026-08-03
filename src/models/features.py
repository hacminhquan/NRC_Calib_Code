"""Architecture-agnostic frozen PyTorch feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class FeatureBatch:
    """Cached frozen-model features, raw model output, and targets."""

    features: np.ndarray
    outputs: np.ndarray
    targets: np.ndarray


def extract_features(model: nn.Module, layer: nn.Module, loader: Iterable[tuple[torch.Tensor, torch.Tensor]], device: torch.device | str, output_transform: Callable[[object], torch.Tensor] | None = None) -> FeatureBatch:
    """Forward a loader once while capturing a named penultimate-layer activation."""
    captured: list[torch.Tensor] = []

    def hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError("The selected feature layer must output a tensor")
        captured.append(output.detach().cpu())

    handle = layer.register_forward_hook(hook)
    outputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    model.eval()
    try:
        with torch.inference_mode():
            for x, y in loader:
                raw = model(x.to(device))
                value = output_transform(raw) if output_transform else raw
                if not isinstance(value, torch.Tensor):
                    raise TypeError("output_transform must return a tensor")
                outputs.append(value.detach().cpu())
                targets.append(y.detach().cpu())
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("Feature extraction received no batches")
    return FeatureBatch(torch.cat(captured).numpy(), torch.cat(outputs).numpy(), torch.cat(targets).numpy())
