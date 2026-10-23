# Running on my laptop GPU

GitHub stores the code and release data. Python uses the GPU on the machine
where it executes. My browser cloud workspace does not automatically gain
access to the laptop GPU. I use a local checkout and local tools for that.

## Current laptop and prerequisites

The reported laptop runs Windows 10 and has an NVIDIA GeForce RTX 2070 with
8 GB VRAM. `nvidia-smi` reports driver 617.42. This establishes driver/device
visibility, not yet PyTorch compatibility or a measured training throughput.

In a normal Windows Command Prompt, check installed prerequisites:

```bat
python --version
git --version
```

Git is needed for the checkout; install it from the official
[Git site](https://git-scm.com/download/win) if missing, then open a new terminal.
Use a supported Python version for the selected PyTorch release; the cloud
project checks currently use Python 3.12.

## Open the project locally

Create a fresh checkout under my user folder, rather than working in a Windows
system directory. If this folder already contains a checkout, inspect its status
before changing branches or pulling; do not overwrite local work.

```bat
cd /d "%USERPROFILE%"
git clone --branch V2 --single-branch https://github.com/MahmoudAsadi97/CGAN_Research.git
cd CGAN_Research
```

All preparation and training commands run from this folder in the local
environment. A local terminal or editor operates on this checkout; it does not
transfer the existing cloud process to the laptop.

## Check PyTorch execution

Create a dedicated environment in the checkout:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

Select a compatible NVIDIA CUDA build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/) after
checking the Python version. Do not run `scripts/setup-cloud.sh` here: it
installs CPU PyTorch for the cloud workspace. Windows and WSL have separate
Python environments; install and check in the one that will run training.

With the intended environment activated:

```bat
python scripts/check_local_gpu.py --device cuda --output gpu-check.json
```

The diagnostic records OS/Python/PyTorch versions, device details, and an actual
float32 forward/backward/Adam update using a tiny synthetic network. It installs
nothing, starts no server, and uploads nothing. It excludes GPU UUIDs and process
lists. Existing reports are preserved; use a new filename for another run.
Local `gpu-check*.json` reports are ignored by Git.

Default `--device auto` requires CUDA/ROCm or MPS and never substitutes CPU.
`--device cpu` explicitly checks CPU execution and does not establish GPU
readiness. Exit code 0 means the requested device passed; exit code 2 means
inspection, execution, or report output failed. Check `gpu_ready` in the report.
Device detection alone is insufficient: a kernel or driver error fails the test.

For other machines, the diagnostic distinguishes ROCm through `torch.version.hip`
even though it uses the `torch.cuda` API. MPS checks use float32 and reject an
enabled `PYTORCH_ENABLE_MPS_FALLBACK=1`; rerun in a new process after removing
that setting. Actual compatibility must be established on the selected hardware.

## Research status

The [dataset audit](research/dataset-audit.md) supplies the actual input schema.
The [implementation plan](research/implementation-plan.md) defines preprocessing,
training, evaluation, and figures. Temperature is optional and does not block
the baseline. The research package remains to be implemented; the environment
check is not a measured-data pilot or evidence of model quality.

Current verification covers CPU execution and failure reporting in the cloud.
No laptop GPU computation or local access has been verified yet. All code
changes stay on `V2`; raw data, checkpoints, and outputs stay outside commits.
