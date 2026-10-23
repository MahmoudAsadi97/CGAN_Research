"""Check readiness reporting, especially cases that could falsely claim a GPU."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_local_gpu.py"


def diagnostic():
    assert SCRIPT.is_file(), "The local GPU diagnostic has not been implemented"
    spec = importlib.util.spec_from_file_location("check_local_gpu", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_cpu_executes_forward_backward_and_optimizer():
    report = diagnostic().collect_report("cpu")
    assert report["ready"] is True
    assert report["gpu_ready"] is False
    assert report["device"] == "cpu"
    assert report["smoke"]["finite_loss"] is True
    assert report["smoke"]["finite_gradients"] is True
    assert report["smoke"]["parameters_changed"] is True


def test_auto_requires_an_accelerator_without_silent_cpu_fallback(monkeypatch):
    module = diagnostic()
    monkeypatch.setattr(module, "available_backends", lambda torch: {"cuda": False, "mps": False})
    report = module.collect_report("auto")
    assert report["ready"] is False and report["gpu_ready"] is False
    assert report["device"] is None
    assert report["error_code"] == "accelerator_unavailable"


def test_explicit_cuda_does_not_switch_to_mps(monkeypatch):
    module = diagnostic()
    monkeypatch.setattr(module, "available_backends", lambda torch: {"cuda": False, "mps": True})
    report = module.collect_report("cuda")
    assert report["ready"] is False
    assert report["error_code"] == "requested_device_unavailable"


def test_missing_torch_returns_actionable_json_report(monkeypatch):
    module = diagnostic()

    def missing():
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(module, "load_torch", missing)
    report = module.collect_report("auto")
    assert report["ready"] is False and report["gpu_ready"] is False
    assert report["error_code"] == "torch_import_failed"
    assert "pytorch.org" in report["next_step"]
    json.dumps(report, allow_nan=False)


def test_driver_or_kernel_failure_is_not_reported_as_gpu_ready(monkeypatch):
    module = diagnostic()
    monkeypatch.setattr(module, "available_backends", lambda torch: {"cuda": True, "mps": False})
    monkeypatch.setattr(module, "device_details", lambda torch, device: {"name": "fixture GPU"})

    def broken_kernel(torch, device):
        raise RuntimeError("no kernel image is available for execution on the device")

    monkeypatch.setattr(module, "run_smoke", broken_kernel)
    report = module.collect_report("cuda")
    assert report["device"] == "cuda"
    assert report["ready"] is False and report["gpu_ready"] is False
    assert report["error_code"] == "device_execution_failed"
    assert "kernel image" in report["error"]


def test_backend_probe_failure_is_a_report_not_an_uncaught_exception(monkeypatch):
    module = diagnostic()

    def broken_probe(torch):
        raise RuntimeError("driver initialization failed")

    monkeypatch.setattr(module, "available_backends", broken_probe)
    report = module.collect_report("auto")
    assert report["ready"] is False
    assert report["error_code"] == "device_probe_failed"


def test_explicit_cpu_check_survives_a_broken_gpu_probe(monkeypatch):
    module = diagnostic()

    def broken_probe(torch):
        raise RuntimeError("driver initialization failed")

    monkeypatch.setattr(module, "available_backends", broken_probe)
    report = module.collect_report("cpu")
    assert report["ready"] is True and report["gpu_ready"] is False
    assert report["smoke"]["parameters_changed"] is True


def test_mps_operator_fallback_is_not_a_strict_gpu_success(monkeypatch):
    module = diagnostic()
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setattr(module, "available_backends", lambda torch: {"cuda": False, "mps": True})
    monkeypatch.setattr(module, "device_details", lambda torch, device: {"backend": "mps"})
    monkeypatch.setattr(module, "run_smoke", lambda torch, device: {"parameters_changed": True})
    report = module.collect_report("mps")
    assert report["ready"] is False and report["gpu_ready"] is False
    assert report["error_code"] == "mps_cpu_fallback_enabled"
    assert report["mps_cpu_fallback_enabled"] is True


def test_cli_cpu_json_and_existing_output_is_preserved(tmp_path):
    diagnostic()
    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--device", "cpu", "--output", str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(output.read_text())
    assert report["ready"] and not report["gpu_ready"]
    assert "hostname" not in report and "username" not in report
    output.write_text("keep this previous report")
    repeated = subprocess.run(
        [sys.executable, str(SCRIPT), "--device", "cpu", "--output", str(output)],
        capture_output=True, text=True, timeout=60,
    )
    assert repeated.returncode != 0
    assert output.read_text() == "keep this previous report"


def test_cli_missing_torch_exits_nonzero_with_json():
    diagnostic()
    # -S skips site packages, providing a real interpreter without torch.
    result = subprocess.run([sys.executable, "-S", str(SCRIPT)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    report = json.loads(result.stdout)
    assert report["error_code"] == "torch_import_failed"
