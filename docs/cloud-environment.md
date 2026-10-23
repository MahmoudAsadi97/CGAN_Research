# CGAN research cloud environment

This branch prepares a CPU research environment. The repository initially had no
application code, data, commits, or main branch. A model architecture and dataset
loader can be added once the research requirements and release dataset are ready.

## Install and activate

The cloud development image provides Python 3.12 and uv. From `/workspace/CGAN_Research`:

```bash
bash scripts/setup-cloud.sh
source /workspace/.cloud-environments/cgan/activate.sh
```

The installer pins all 125 Python packages and uses official PyPI and PyTorch CPU
packages with TLS and artifact verification enabled. It installs into
`/workspace/.cloud-environments/cgan/venv`; caches, notebook configuration, and
validation helpers also remain outside the repository. It does not modify the
system Python environment. The same installer is saved in the cloud environment
configuration for restoring or refreshing this setup.

The baseline includes PyTorch 2.14.1 CPU, torchvision 0.29.1 CPU, NumPy, pandas,
SciPy, scikit-learn, Pillow, Matplotlib, TensorBoard, JupyterLab, ipykernel,
nbclient, pytest, and tqdm. Future dependency changes should be deliberate and
followed by the checks below.

## Validate

After activation:

```bash
uv pip check --python /workspace/.cloud-environments/cgan/venv/bin/python
python /workspace/.cloud-environments/cgan/smoke.py
```

The smoke check trains small synthetic conditional generator/discriminator
networks for three steps. It asserts finite losses and gradients, parameter
updates, checkpoint save/load, torchvision image transforms and native NMS,
scientific Python operations, PNG output, TensorBoard event read/write, and
execution through the registered notebook kernel. These are environment checks;
they do not establish model quality or validate the forthcoming dataset.

## Notebooks

After activation, run this from `/workspace/CGAN_Research` when needed:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888 --ServerApp.port_retries=0 --ServerApp.root_dir=/workspace/CGAN_Research
```

Use the `Python (CGAN CPU)` kernel. Jupyter authentication stays enabled. The
onboarding UI does not expose a notebook preview; the kernel also supports
programmatic notebook execution. Processes must be restarted after restoring a
cloud environment snapshot.

## Dataset and GPU training

The `Dataset` release now includes the audited `raw.zip`; its schema, source
checksum, coverage, and provider indicator definitions are recorded in the
[dataset audit](research/dataset-audit.md). Cloud data is stored outside the
checkout in `/workspace/datasets/CGAN_Research`. Keep large datasets, checkpoints,
and generated training outputs out of Git commits.

GitHub hosts the code and release files. Training runs on a compute machine.
This cloud development environment has no NVIDIA GPU. A laptop GPU can run the
project through a local checkout; see [laptop GPU setup](local-gpu.md).
Colab, Kaggle, or a GPU cloud VM are other runtime options. Use the chosen
runtime's compatible GPU PyTorch installation instead of running this
CPU-specific installer there. A browser cloud session does not attach to the
laptop GPU automatically.

The cloud workspace is already isolated. I use the provided checkout and do not
need an additional Git worktree for this project.
