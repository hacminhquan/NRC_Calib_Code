"""Colab-aware runtime initialization used at the start of every notebook."""
from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeInfo:
    """Resolved project and hardware configuration."""

    project_root: Path
    checkpoint_root: Path
    cuda_available: bool
    torch_version: str
    cuda_version: str | None
    ram_gib: float
    gpu_name: str | None


def configure_runtime(project_root: str | Path | None = None) -> RuntimeInfo:
    """Resolve the uploaded/local project root and enable safe CUDA settings."""
    import psutil
    import torch

    configured = project_root or os.environ.get("NRC_CAL_PROJECT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([Path("/content/NRC_CALIB_CODE"), Path.cwd().parent, Path.cwd()])
    root = next((path for path in candidates if (path / "src").is_dir()), None)
    if root is None:
        raise FileNotFoundError(
            "NRC-Cal project not found. Upload the project ZIP in notebook 00 or set "
            "NRC_CAL_PROJECT_ROOT to a directory containing src/."
        )
    root = root.resolve()
    for name in ("outputs", "figures", "checkpoints", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    os.environ["NRC_CAL_PROJECT_ROOT"] = str(root)
    os.environ["NRC_CAL_CHECKPOINT_ROOT"] = str(checkpoint_root)
    cuda = torch.cuda.is_available()
    gpu_name = None
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        gpu_name = torch.cuda.get_device_name(0)
    info = RuntimeInfo(root, checkpoint_root, cuda, torch.__version__, torch.version.cuda,
                       psutil.virtual_memory().total / 2**30, gpu_name)
    logging.getLogger(__name__).info("Runtime: %s", asdict(info))
    return info


def add_project_source(project_root: Path) -> None:
    """Place the project's helper modules first on Python's import path."""
    source = str(project_root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def gpu_summary() -> str:
    """Return a compact CUDA-driver summary without failing on CPU hosts."""
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi unavailable"
    return subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        text=True, capture_output=True, check=False,
    ).stdout.strip()
