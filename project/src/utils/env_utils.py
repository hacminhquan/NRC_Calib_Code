"""Environment bootstrap utilities for the NRC-Cal project.

This module backs ``notebooks/00_environment.ipynb``. It intentionally
contains **no NRC-specific mathematics** — its only job is to make the
notebook environment (Colab or local), the project directory layout, and
the Python/CUDA stack verifiable and reproducible.

Design notes
------------
- Every function is safe to call outside Colab and without a GPU: it
  degrades gracefully and reports what it found, rather than raising.
- Heavy/optional dependencies (torch, psutil) are imported lazily inside
  the functions that need them, so importing this module never requires
  the full project dependency stack to already be installed.
- All functions are pure with respect to global state except where the
  side effect is the explicit point of the function (mounting Drive,
  creating directories, cloning a repo) — those are named accordingly
  and documented.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("nrc_cal.env_utils")
if not logger.handlers:
    # Keep this idempotent: re-importing the module (e.g. notebook re-run)
    # must not attach duplicate handlers.
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Colab / runtime detection
# --------------------------------------------------------------------------- #


def is_colab() -> bool:
    """Return True iff the current runtime is Google Colab.

    Returns
    -------
    bool
        True if the ``google.colab`` module is importable in this runtime.
    """
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


# --------------------------------------------------------------------------- #
# Option A — Google Drive
# --------------------------------------------------------------------------- #


def mount_google_drive(mount_point: str = "/content/drive") -> Optional[Path]:
    """Mount Google Drive if running in Colab.

    Parameters
    ----------
    mount_point:
        Local path Colab will mount Drive under.

    Returns
    -------
    Optional[Path]
        Path to ``MyDrive`` inside the mounted Drive, or ``None`` if not
        running in Colab (Drive mounting is a Colab-only operation).

    Raises
    ------
    RuntimeError
        If running in Colab but the mount call itself fails.
    """
    if not is_colab():
        logger.info("Not running in Colab — skipping Google Drive mount.")
        return None

    from google.colab import drive  # type: ignore[import-not-found]

    try:
        drive.mount(mount_point, force_remount=False)
    except Exception as exc:  # pragma: no cover - depends on live Colab session
        raise RuntimeError(f"Failed to mount Google Drive at {mount_point}: {exc}") from exc

    my_drive = Path(mount_point) / "MyDrive"
    if not my_drive.exists():
        raise RuntimeError(
            f"Drive mounted at {mount_point} but expected subpath {my_drive} "
            "does not exist."
        )
    logger.info("Google Drive mounted at %s", my_drive)
    return my_drive


# --------------------------------------------------------------------------- #
# Option B — GitHub
# --------------------------------------------------------------------------- #


def clone_or_pull_repo(repo_url: str, dest: Path, branch: str = "main") -> bool:
    """Clone ``repo_url`` into ``dest`` if absent, else ``git pull`` it.

    Generic on purpose: reused here for the user's own project repo
    (Option B) and later, unmodified, by notebook 01 for the two Dheur
    & Ben Taieb repositories.

    Parameters
    ----------
    repo_url:
        HTTPS git URL. If empty, this is a no-op (returns False).
    dest:
        Local destination directory.
    branch:
        Branch to check out / pull.

    Returns
    -------
    bool
        True if a clone or pull was actually performed, False if skipped
        (empty URL) or if the operation failed (logged, not raised — a
        missing remote should not crash the whole notebook run).
    """
    if not repo_url:
        logger.info("No repo_url configured — skipping clone/pull for %s.", dest)
        return False

    dest = Path(dest)
    try:
        if dest.exists() and (dest / ".git").exists():
            logger.info("Repo already present at %s — pulling latest %s.", dest, branch)
            subprocess.run(
                ["git", "-C", str(dest), "pull", "origin", branch],
                check=True, capture_output=True, text=True,
            )
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Cloning %s (branch=%s) into %s.", repo_url, branch, dest)
            subprocess.run(
                ["git", "clone", "--branch", branch, "--single-branch", repo_url, str(dest)],
                check=True, capture_output=True, text=True,
            )
        return True
    except subprocess.CalledProcessError as exc:
        logger.error(
            "git operation failed for %s: %s", repo_url, exc.stderr.strip() if exc.stderr else exc
        )
        return False


# --------------------------------------------------------------------------- #
# Option C — SSH / rsync
# --------------------------------------------------------------------------- #


def build_rsync_command(
    remote_user: str,
    remote_host: str,
    remote_path: str,
    local_path: Path,
    ssh_port: int = 22,
) -> str:
    """Build (but do not execute) an rsync-over-SSH pull command.

    Important
    ---------
    This function only *constructs* the command string. Running it will
    only succeed if ``remote_host`` is actually reachable from wherever
    this code executes. Google Colab runs on Google's cloud network and
    **cannot** reach a MacBook sitting behind typical home NAT without an
    intermediary (Tailscale, a reverse SSH tunnel, ngrok, or a public
    static IP with port forwarding). That is a networking prerequisite,
    not something this function can paper over — see README.md.

    Parameters
    ----------
    remote_user, remote_host, remote_path:
        SSH target identifying the source folder on the MacBook.
    local_path:
        Local (Colab-side) destination directory.
    ssh_port:
        SSH port to connect on.

    Returns
    -------
    str
        A ready-to-run shell command.
    """
    local_path = Path(local_path)
    return (
        f"rsync -avz -e 'ssh -p {ssh_port}' "
        f"{remote_user}@{remote_host}:{remote_path}/ {local_path}/"
    )


def run_rsync(command: str, timeout_s: int = 15) -> Dict[str, Any]:
    """Execute an rsync command produced by :func:`build_rsync_command`.

    Uses a short default timeout deliberately: if the remote host is
    unreachable (the common case without a tunnel), this fails fast with
    a clear message instead of hanging the notebook.

    Returns
    -------
    Dict[str, Any]
        ``{"success": bool, "stdout": str, "stderr": str}``.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout_s
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        msg = (
            f"rsync timed out after {timeout_s}s — the remote host is most likely "
            "unreachable from this network. See README.md, Option C."
        )
        logger.warning(msg)
        return {"success": False, "stdout": "", "stderr": msg}


# --------------------------------------------------------------------------- #
# Project root resolution & directory scaffolding
# --------------------------------------------------------------------------- #

_EXPECTED_SUBDIRS = (
    "notebooks", "src", "configs", "outputs", "figures", "checkpoints", "logs", "tests",
)


def locate_project_root(
    drive_root: Optional[Path],
    repo_dir_name: str = "project",
    local_fallback: Optional[Path] = None,
) -> Path:
    """Resolve the single project root directory to use for this session.

    Resolution order:

    1. If Drive is mounted (``drive_root`` given) and
       ``drive_root / repo_dir_name`` exists → use it (Option A).
    2. Else if ``local_fallback`` is given → use it (running locally / repo
       already checked out here, e.g. via Option B or C beforehand).
    3. Else → fall back to the current working directory.

    In all cases, the resolved directory is created if it does not yet
    exist; this function never raises for a missing directory, only for
    a path that exists but is not a directory.

    Parameters
    ----------
    drive_root:
        Result of :func:`mount_google_drive`, or None.
    repo_dir_name:
        Folder name to look for / create under Drive.
    local_fallback:
        Explicit local path to prefer when Drive is unavailable.

    Returns
    -------
    Path
        The resolved project root.
    """
    candidate: Path
    if drive_root is not None:
        candidate = drive_root / repo_dir_name
        logger.info("Using Google Drive project root: %s", candidate)
    elif local_fallback is not None:
        candidate = Path(local_fallback)
        logger.info("Using local fallback project root: %s", candidate)
    else:
        candidate = Path.cwd()
        logger.info("No Drive or local_fallback given — using cwd: %s", candidate)

    if candidate.exists() and not candidate.is_dir():
        raise NotADirectoryError(f"Resolved project root {candidate} exists but is not a directory.")

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def ensure_project_structure(project_root: Path) -> Dict[str, Path]:
    """Create the expected subdirectory layout under ``project_root`` if missing.

    Parameters
    ----------
    project_root:
        Root directory (already resolved by :func:`locate_project_root`).

    Returns
    -------
    Dict[str, Path]
        Mapping from subdirectory name to its absolute path, for the
        names listed in ``_EXPECTED_SUBDIRS``.
    """
    project_root = Path(project_root)
    paths: Dict[str, Path] = {}
    for name in _EXPECTED_SUBDIRS:
        sub = project_root / name
        sub.mkdir(parents=True, exist_ok=True)
        paths[name] = sub
    logger.info("Verified project structure under %s (%d subdirs).", project_root, len(paths))
    return paths


# --------------------------------------------------------------------------- #
# PyTorch / CUDA verification
# --------------------------------------------------------------------------- #


def verify_pytorch_cuda() -> Dict[str, Any]:
    """Report the PyTorch/CUDA stack available in this runtime.

    Returns
    -------
    Dict[str, Any]
        Always contains ``"torch_installed": bool``. If True, also
        contains ``torch_version``, ``cuda_available``, ``cuda_version``,
        ``device_count``, and ``device_name`` (the last two only if a
        CUDA device is available). If torch is not installed, the dict
        contains only the False flag plus an explanatory ``"note"`` — this
        function never raises for a missing/absent GPU, since notebook 00
        does not require one.
    """
    try:
        import torch
    except ImportError:
        return {
            "torch_installed": False,
            "note": "PyTorch is not installed in this environment yet — "
                    "run the dependency install cell first.",
        }

    report: Dict[str, Any] = {
        "torch_installed": True,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if report["cuda_available"]:
        report["cuda_version"] = torch.version.cuda
        report["device_count"] = torch.cuda.device_count()
        report["device_name"] = torch.cuda.get_device_name(0)
    else:
        report["cuda_version"] = None
        report["device_count"] = 0
        report["device_name"] = None
    return report


def verify_gpu_matches_target(target_substring: str = "T4") -> Dict[str, Any]:
    """Check whether the attached GPU (if any) matches the intended target.

    This never raises and never blocks execution — Colab's GPU assignment
    depends on live availability and the user's Colab tier, so a mismatch
    is reported as a warning, not an error. The caller decides whether to
    act on it (e.g. by requesting Runtime > Change runtime type again).

    Parameters
    ----------
    target_substring:
        Case-insensitive substring expected in the CUDA device name
        (e.g. "T4" for an NVIDIA T4).

    Returns
    -------
    Dict[str, Any]
        ``{"has_gpu": bool, "device_name": Optional[str], "matches_target": bool}``.
    """
    pytorch_info = verify_pytorch_cuda()
    if not pytorch_info.get("torch_installed") or not pytorch_info.get("cuda_available"):
        return {"has_gpu": False, "device_name": None, "matches_target": False}

    device_name = pytorch_info.get("device_name") or ""
    matches = target_substring.lower() in device_name.lower()
    if not matches:
        logger.warning(
            "Attached GPU is '%s', which does not match the intended target '%s'. "
            "This is not fatal, but batch sizes / mixed-precision choices tuned for "
            "%s may not be optimal here. In Colab: Runtime > Change runtime type to retry.",
            device_name, target_substring, target_substring,
        )
    else:
        logger.info("Attached GPU '%s' matches target '%s'.", device_name, target_substring)
    return {"has_gpu": True, "device_name": device_name, "matches_target": matches}


def enable_mixed_precision_defaults() -> Dict[str, Any]:
    """Set sane default matmul precision for mixed-precision training.

    This only *configures* PyTorch defaults (``float32_matmul_precision``);
    it does not itself wrap any training loop in autocast — that happens
    in the training/calibration notebooks that actually run models.

    Returns
    -------
    Dict[str, Any]
        What was set, or a note if torch/CUDA are unavailable.
    """
    try:
        import torch
    except ImportError:
        return {"applied": False, "note": "PyTorch not installed."}

    if not torch.cuda.is_available():
        return {"applied": False, "note": "No CUDA device — mixed precision not applicable."}

    torch.set_float32_matmul_precision("high")
    return {"applied": True, "float32_matmul_precision": "high"}


# --------------------------------------------------------------------------- #
# System resource detection
# --------------------------------------------------------------------------- #


def detect_system_resources() -> Dict[str, Any]:
    """Report RAM and disk space available in this runtime.

    Uses ``psutil`` when available; falls back to ``shutil.disk_usage``
    for disk space alone if ``psutil`` is not installed, so this function
    still returns something useful rather than failing outright.

    Returns
    -------
    Dict[str, Any]
        Keys: ``ram_total_gb``, ``ram_available_gb`` (None if psutil
        missing), ``disk_total_gb``, ``disk_free_gb``.
    """
    import shutil

    report: Dict[str, Any] = {}
    try:
        import psutil

        vm = psutil.virtual_memory()
        report["ram_total_gb"] = round(vm.total / 1e9, 2)
        report["ram_available_gb"] = round(vm.available / 1e9, 2)
    except ImportError:
        logger.warning("psutil not installed — RAM detection skipped (disk info still reported).")
        report["ram_total_gb"] = None
        report["ram_available_gb"] = None

    usage = shutil.disk_usage(Path.cwd())
    report["disk_total_gb"] = round(usage.total / 1e9, 2)
    report["disk_free_gb"] = round(usage.free / 1e9, 2)
    return report


# --------------------------------------------------------------------------- #
# Full environment report
# --------------------------------------------------------------------------- #


@dataclass
class EnvironmentReport:
    """Complete snapshot of the notebook 00 environment checks."""

    in_colab: bool
    project_root: str
    pytorch: Dict[str, Any] = field(default_factory=dict)
    mixed_precision: Dict[str, Any] = field(default_factory=dict)
    system: Dict[str, Any] = field(default_factory=dict)
    python_version: str = field(default_factory=lambda: sys.version)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_environment_report(project_root: Path) -> EnvironmentReport:
    """Assemble the full environment report for this session.

    Parameters
    ----------
    project_root:
        Resolved project root (for record-keeping in the report only).

    Returns
    -------
    EnvironmentReport
    """
    return EnvironmentReport(
        in_colab=is_colab(),
        project_root=str(project_root),
        pytorch=verify_pytorch_cuda(),
        mixed_precision=enable_mixed_precision_defaults(),
        system=detect_system_resources(),
    )


def save_environment_report(report: EnvironmentReport, path: Path) -> Path:
    """Serialize an :class:`EnvironmentReport` to JSON at ``path``.

    Parameters
    ----------
    report:
        The report to save.
    path:
        Destination file path (parent directories are created if needed).

    Returns
    -------
    Path
        The path written to (same as input, returned for chaining).

    Raises
    ------
    OSError
        If the file cannot be written (e.g. read-only filesystem).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(report.to_dict(), indent=2))
    except OSError as exc:
        raise OSError(f"Could not write environment report to {path}: {exc}") from exc
    logger.info("Environment report saved to %s", path)
    return path


def pretty_print_report(report: EnvironmentReport) -> None:
    """Print ``report`` as a readable table, using ``rich`` if available."""
    data = report.to_dict()
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="NRC-Cal — Environment Report")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="white")
        for section, value in data.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    table.add_row(f"{section}.{k}", str(v))
            else:
                table.add_row(section, str(value))
        console.print(table)
    except ImportError:
        # Plain fallback — never let a missing optional dependency hide
        # the actual environment info from the user.
        print(json.dumps(data, indent=2))
