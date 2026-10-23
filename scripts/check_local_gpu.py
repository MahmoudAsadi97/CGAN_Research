"""Check local PyTorch acceleration with a small synthetic optimizer step.

Run on the machine that will train the model:
    python scripts/check_local_gpu.py --output gpu-check.json

Default device 'auto' requires CUDA/ROCm or Apple MPS; it never falls back to
CPU. '--device cpu' explicitly checks CPU execution. No packages or drivers are
installed, no server is started, and nothing is uploaded. Exit 0 means the
requested device passed the smoke check; exit 2 means it did not. A successful
CPU check does not establish GPU readiness or research model quality.
"""

import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def load_torch():
    return importlib.import_module("torch")


def available_backends(torch):
    mps = getattr(torch.backends, "mps", None)
    return {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(mps is not None and mps.is_available()),
    }


def nvidia_driver_summary():
    """Optional vendor tool; omit process lists, UUIDs, and system identifiers."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().splitlines() if result.returncode == 0 else None


def device_details(torch, device):
    if device == "cuda":
        properties = torch.cuda.get_device_properties(0)
        return {
            "name": properties.name,
            "index": 0,
            "visible_device_count": torch.cuda.device_count(),
            "total_memory_bytes": properties.total_memory,
            "backend": "rocm" if getattr(torch.version, "hip", None) else "cuda",
        }
    if device == "mps":
        return {"backend": "mps", "name": "Apple Metal device"}
    return {"backend": "cpu", "name": "CPU"}


def run_smoke(torch, device):
    """Exercise float32 kernels, autograd, and Adam; this is not the research GAN."""
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64), torch.nn.LeakyReLU(0.2), torch.nn.Linear(64, 24)
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    inputs = torch.ones((8, 32), device=device, dtype=torch.float32)
    initial = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = prediction.square().mean()
    finite_loss = bool(torch.isfinite(loss).item())
    if not finite_loss:
        raise RuntimeError("The synthetic forward pass produced a nonfinite loss")
    loss.backward()
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all().item())
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("The synthetic backward pass produced missing or nonfinite gradients")
    optimizer.step()
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    changed = any(
        not torch.equal(before, after.detach())
        for before, after in zip(initial, model.parameters())
    )
    finite_parameters = all(bool(torch.isfinite(p).all().item()) for p in model.parameters())
    if not changed or not finite_parameters:
        raise RuntimeError("Adam failed to produce a finite parameter update")
    return {
        "kind": "synthetic_environment_check",
        "dtype": "float32",
        "optimizer_steps": 1,
        "finite_loss": finite_loss,
        "finite_gradients": finite_gradients,
        "parameters_changed": changed,
        "finite_parameters": finite_parameters,
    }


def collect_report(requested="auto"):
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("Device must be auto, cuda, mps, or cpu")
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "requested_device": requested,
        "device": None,
        "ready": False,
        "gpu_ready": False,
        "mps_cpu_fallback_enabled": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1",
        "nvidia_driver_summary": nvidia_driver_summary(),
        "research_training_executed": False,
    }
    try:
        torch = load_torch()
        report["torch"] = {
            "version": str(torch.__version__),
            "cuda_build": getattr(torch.version, "cuda", None),
            "rocm_build": getattr(torch.version, "hip", None),
        }
    except Exception as error:
        report.update(
            error_code="torch_import_failed", error=str(error),
            next_step="Install a compatible PyTorch build in this Python environment using https://pytorch.org/get-started/locally/ and rerun. Do not use the repository's CPU cloud installer for GPU setup.",
        )
        return report
    try:
        # A broken accelerator driver must not block an explicit CPU check.
        available = available_backends(torch) if requested != "cpu" else None
        report["available_backends"] = available
    except Exception as error:
        report.update(error_code="device_probe_failed", error=str(error),
                      next_step="Check the GPU driver and PyTorch compatibility in this terminal environment.")
        return report
    device = requested
    if requested == "auto":
        device = "cuda" if available["cuda"] else "mps" if available["mps"] else None
        if device is None:
            report.update(error_code="accelerator_unavailable",
                          next_step="No accelerator is available to this PyTorch installation. Check the GPU, driver, and installed build. Use --device cpu only for an explicit CPU check.")
            return report
    elif requested != "cpu" and not available[requested]:
        report.update(error_code="requested_device_unavailable",
                      next_step=f"The requested {requested} backend is unavailable. Check support for this GPU and installed PyTorch build.")
        return report
    report["device"] = device
    if device == "mps" and report["mps_cpu_fallback_enabled"]:
        report.update(error_code="mps_cpu_fallback_enabled",
                      next_step="Unset PYTORCH_ENABLE_MPS_FALLBACK and rerun in a new Python process. This strict check cannot establish GPU-only operator support while CPU fallback is enabled.")
        return report
    try:
        report["device_details"] = device_details(torch, device)
        report["smoke"] = run_smoke(torch, device)
    except Exception as error:
        report.update(error_code="device_execution_failed", error=str(error),
                      next_step="Device detection succeeded but computation failed. Check driver, GPU architecture/build compatibility, available memory, and backend support before training.")
        return report
    report["ready"] = True
    report["gpu_ready"] = device in {"cuda", "mps"}
    report["next_step"] = "Environment smoke check passed. Prepare and test the research pipeline before starting a bounded measured-data pilot."
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, help="Optional new JSON report; existing files are never overwritten.")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}. Choose a new filename.")
    report = collect_report(args.device)
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        except OSError as error:
            parser.error(f"Cannot create output: {error}")
    sys.stdout.write(payload)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
