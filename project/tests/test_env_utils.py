"""Unit tests for src.utils.env_utils.

These tests intentionally avoid requiring an actual Colab runtime or GPU:
notebook 00 must be verifiable in plain CI / a developer's laptop too.
Where a function's behavior legitimately depends on optional heavy deps
(torch, psutil), the tests assert the *contract* (correct keys, correct
types, graceful fallback) rather than a specific hardware outcome.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import env_utils  # noqa: E402


def test_is_colab_false_outside_colab():
    """Outside an actual Colab runtime, this must be False, not raise."""
    assert env_utils.is_colab() is False


def test_mount_google_drive_noop_outside_colab():
    """Drive mounting must be a safe no-op (returns None) outside Colab."""
    assert env_utils.mount_google_drive() is None


def test_locate_project_root_creates_local_fallback(tmp_path):
    fallback = tmp_path / "my_project"
    root = env_utils.locate_project_root(drive_root=None, local_fallback=fallback)
    assert root == fallback
    assert root.exists() and root.is_dir()


def test_locate_project_root_prefers_drive_when_given(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    fallback = tmp_path / "should_not_be_used"
    root = env_utils.locate_project_root(
        drive_root=drive_root, repo_dir_name="project", local_fallback=fallback
    )
    assert root == drive_root / "project"
    assert not fallback.exists()


def test_locate_project_root_rejects_file_at_path(tmp_path):
    bad_path = tmp_path / "not_a_dir"
    bad_path.write_text("i am a file")
    with pytest.raises(NotADirectoryError):
        env_utils.locate_project_root(drive_root=None, local_fallback=bad_path)


def test_ensure_project_structure_creates_all_expected_subdirs(tmp_path):
    paths = env_utils.ensure_project_structure(tmp_path)
    expected = {
        "notebooks", "src", "configs", "outputs",
        "figures", "checkpoints", "logs", "tests",
    }
    assert set(paths.keys()) == expected
    for p in paths.values():
        assert p.exists() and p.is_dir()


def test_ensure_project_structure_is_idempotent(tmp_path):
    first = env_utils.ensure_project_structure(tmp_path)
    second = env_utils.ensure_project_structure(tmp_path)
    assert first.keys() == second.keys()


def test_clone_or_pull_repo_noop_on_empty_url(tmp_path):
    result = env_utils.clone_or_pull_repo(repo_url="", dest=tmp_path / "repo")
    assert result is False
    assert not (tmp_path / "repo").exists()


def test_build_rsync_command_format():
    cmd = env_utils.build_rsync_command(
        remote_user="quan",
        remote_host="my-mac.tailnet.ts.net",
        remote_path="/Users/quan/project",
        local_path=Path("/content/project"),
        ssh_port=2222,
    )
    assert cmd.startswith("rsync -avz -e 'ssh -p 2222'")
    assert "quan@my-mac.tailnet.ts.net:/Users/quan/project/" in cmd
    assert cmd.endswith("/content/project/")


def test_run_rsync_times_out_gracefully_on_unreachable_host():
    """Regression test for the documented networking caveat: an
    unreachable host must fail fast (short timeout) with success=False,
    never hang the notebook."""
    cmd = env_utils.build_rsync_command(
        remote_user="nobody",
        remote_host="10.255.255.1",  # non-routable — guaranteed unreachable
        remote_path="/tmp",
        local_path=Path("/tmp/nrc_cal_rsync_test"),
    )
    result = env_utils.run_rsync(cmd, timeout_s=2)
    assert result["success"] is False


def test_verify_pytorch_cuda_contract():
    report = env_utils.verify_pytorch_cuda()
    assert "torch_installed" in report
    if report["torch_installed"]:
        for key in ("torch_version", "cuda_available", "cuda_version", "device_count", "device_name"):
            assert key in report
    else:
        assert "note" in report


def test_verify_gpu_matches_target_contract():
    result = env_utils.verify_gpu_matches_target(target_substring="T4")
    assert "has_gpu" in result and "device_name" in result and "matches_target" in result
    if not result["has_gpu"]:
        assert result["matches_target"] is False
        assert result["device_name"] is None


def test_detect_system_resources_contract():
    report = env_utils.detect_system_resources()
    for key in ("ram_total_gb", "ram_available_gb", "disk_total_gb", "disk_free_gb"):
        assert key in report
    # Disk info must always be populated regardless of psutil availability.
    assert report["disk_total_gb"] > 0
    assert report["disk_free_gb"] >= 0


def test_build_environment_report_and_save(tmp_path):
    report = env_utils.build_environment_report(project_root=tmp_path)
    assert isinstance(report, env_utils.EnvironmentReport)
    assert report.project_root == str(tmp_path)

    out_path = tmp_path / "logs" / "environment_report.json"
    saved_path = env_utils.save_environment_report(report, out_path)
    assert saved_path == out_path
    assert out_path.exists()

    loaded = json.loads(out_path.read_text())
    assert loaded["in_colab"] is False
    assert "pytorch" in loaded and "system" in loaded


def test_pretty_print_report_does_not_raise(capsys):
    report = env_utils.build_environment_report(project_root=Path("/tmp"))
    env_utils.pretty_print_report(report)
    captured = capsys.readouterr()
    assert len(captured.out) > 0
