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
    """Mount Drive when available, resolve roots, and enable safe CUDA settings."""
    import psutil
    import torch

    drive_root: Path | None = None
    try:
        from google.colab import drive  # type: ignore

        drive_root = Path("/content/drive")
        drive.mount(str(drive_root), force_remount=False)
    except ImportError:
        pass
    configured = project_root or os.environ.get("NRC_CAL_PROJECT_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    if drive_root is not None:
        candidates.extend([drive_root / "MyDrive" / "NRC-Cal" / "project", drive_root / "MyDrive" / "project"])
    candidates.extend([Path.cwd().parent, Path.cwd(), Path("/content/NRC-Cal/project")])
    root = next((path for path in candidates if (path / "src").is_dir()), candidates[0])
    root = root.resolve()
    for name in ("outputs", "figures", "checkpoints", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    if drive_root is not None and not str(root).startswith(str(drive_root)):
        checkpoint_root = drive_root / "MyDrive" / "NRC-Cal" / "project" / "checkpoints"
    else:
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
