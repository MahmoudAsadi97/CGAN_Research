#!/usr/bin/env bash
# Reproduce the validated CPU research environment without changing project manifests.
set -euo pipefail
cgan_root=/workspace/.cloud-environments/cgan
export UV_CACHE_DIR=/workspace/.cloud-environments/uv-cache
mkdir -p "$cgan_root"
if [[ ! -x "$cgan_root/venv/bin/python" ]]; then
    uv venv --python python3.12 --no-python-downloads "$cgan_root/venv"
fi
"$cgan_root/venv/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'
cat > "$cgan_root/requirements.txt" <<'CGAN_REQUIREMENTS'
absl-py==2.5.1
anyio==4.15.1
argon2-cffi==25.1.0
argon2-cffi-bindings==26.1.0
arrow==1.4.0
asttokens==3.0.2
async-lru==2.4.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bleach==6.4.0
certifi==2026.10.19
cffi==2.1.1
charset-normalizer==3.5.2
cloudpickle==3.1.2
comm==0.2.3
contourpy==1.4.0
cycler==0.12.1
debugpy==1.8.22
defusedxml==0.7.1
executing==2.3.0
fastjsonschema==2.22.2
filelock==4.2.4
fonttools==4.67.0
fqdn==1.6.0
fsspec==2026.9.0
grpcio==1.84.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
idna==3.20
iniconfig==2.3.1
ipykernel==7.4.0
ipython==9.17.1
ipython-pygments-lexers==1.1.1
isoduration==20.11.0
jedi==0.20.1
jinja2==3.1.6
joblib==1.6.0
json5==0.17.3
jsonpointer==3.4
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter-builder==1.2.3
jupyter-client==8.10.0
jupyter-core==5.9.1
jupyter-events==0.12.1
jupyter-lsp==2.3.1
jupyter-server==2.21.1
jupyter-server-terminals==0.5.4
jupyterlab==4.6.4
jupyterlab-pygments==0.3.0
jupyterlab-server==2.28.1
kiwisolver==1.5.1
lark==1.3.1
markdown==3.11.2
markupsafe==3.0.4
matplotlib==3.11.2
matplotlib-inline==0.2.2
mistune==3.4.0
mpmath==1.3.0
narwhals==2.27.1
nbclient==0.11.0
nbconvert==7.17.2
nbformat==5.11.1
nest-asyncio2==1.7.4
networkx==3.7
notebook-shim==0.2.4
numpy==2.5.4
packaging==26.3
pandas==2.3.3
pandocfilters==1.5.1
parso==0.8.7
pexpect==4.9.0
pillow==12.3.0
platformdirs==4.13.5
pluggy==1.7.0
prometheus-client==0.26.0
prompt-toolkit==3.0.53
protobuf==7.36.2
psutil==7.2.2
ptyprocess==0.7.0
pure-eval==0.2.4
pycparser==3.11
pygments==2.21.0
pyparsing==3.3.3
pytest==9.1.1
python-dateutil==2.9.0.post0
python-json-logger==4.2.0
pytz==2026.5
pyyaml==6.0.3
pyzmq==27.2.0
referencing==0.37.0
requests==2.34.2
rfc3339-validator==0.1.4
rfc3986-validator==0.1.1
rfc3987-syntax==1.1.0
rpds-py==2026.9.1
scikit-learn==1.9.1
scipy==1.18.1
send2trash==2.1.0
setuptools==84.2.1
six==1.17.0
soupsieve==3.0.3
stack-data==0.6.3
sympy==1.14.0
tensorboard==2.21.0
tensorboard-data-server==0.7.2
terminado==0.18.1
threadpoolctl==3.7.0
tinycss2==1.5.1
torch==2.14.1+cpu
torchvision==0.29.1+cpu
tornado==6.5.10
tqdm==4.70.1
traitlets==5.16.1
typing-extensions==4.16.0
tzdata==2026.5
uri-template==1.3.0
urllib3==2.8.0
wcwidth==0.9.3
webcolors==25.10.0
webencodings==0.6.1
websocket-client==1.9.2
werkzeug==3.1.9
CGAN_REQUIREMENTS
uv pip install --python "$cgan_root/venv/bin/python" --torch-backend cpu --only-binary :all: --strict -r "$cgan_root/requirements.txt"
cat > "$cgan_root/activate.sh" <<'CGAN_ACTIVATE'
# Source this file to activate the managed cloud research environment.
source /workspace/.cloud-environments/cgan/venv/bin/activate
export UV_CACHE_DIR=/workspace/.cloud-environments/uv-cache
export JUPYTER_CONFIG_DIR=/workspace/.cloud-environments/cgan/jupyter/config
export JUPYTER_DATA_DIR=/workspace/.cloud-environments/cgan/jupyter/data
export JUPYTER_RUNTIME_DIR=/workspace/.cloud-environments/cgan/jupyter/runtime
export IPYTHONDIR=/workspace/.cloud-environments/cgan/ipython
export MPLCONFIGDIR=/workspace/.cloud-environments/cgan/matplotlib
export XDG_CACHE_HOME=/workspace/.cloud-environments/cgan/cache
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
CGAN_ACTIVATE
source "$cgan_root/activate.sh"
mkdir -p "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
chmod 700 "$JUPYTER_RUNTIME_DIR"
python -m ipykernel install --prefix "$cgan_root/venv" --name cgan-research --display-name 'Python (CGAN CPU)'
cat > "$cgan_root/smoke.py" <<'CGAN_SMOKE'
"""Validate the cloud research dependencies using synthetic data only."""
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torchvision
from nbclient import NotebookClient
from sklearn.metrics import accuracy_score
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.ops import nms


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    labels = torch.eye(2).repeat(4, 1)
    generator = nn.Sequential(nn.Linear(10, 32), nn.LeakyReLU(0.2), nn.Linear(32, 16), nn.Tanh())
    discriminator = nn.Sequential(nn.Linear(18, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    initial_gen = [p.detach().clone() for p in generator.parameters()]
    initial_disc = [p.detach().clone() for p in discriminator.parameters()]
    for _ in range(3):
        real = torch.randn(8, 16).tanh()
        fake = generator(torch.cat([torch.randn(8, 8), labels], dim=1))
        disc_optimizer.zero_grad()
        disc_loss = criterion(discriminator(torch.cat([real, labels], dim=1)), torch.ones(8, 1))
        disc_loss += criterion(discriminator(torch.cat([fake.detach(), labels], dim=1)), torch.zeros(8, 1))
        disc_loss.backward()
        disc_optimizer.step()
        gen_optimizer.zero_grad()
        gen_loss = criterion(discriminator(torch.cat([fake, labels], dim=1)), torch.ones(8, 1))
        gen_loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in generator.parameters())
        gen_optimizer.step()
        assert torch.isfinite(disc_loss) and torch.isfinite(gen_loss)
    assert any(not torch.equal(a, b) for a, b in zip(initial_gen, generator.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(initial_disc, discriminator.parameters()))
    assert fake.shape == (8, 16)
    image = transforms.ToPILImage()(torch.rand(3, 16, 16))
    assert transforms.ToTensor()(image).shape == (3, 16, 16)
    assert nms(torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]]), torch.tensor([0.9, 0.8]), 0.5).tolist() == [0]
    assert np.isfinite(fake.detach().numpy()).all()
    assert pd.DataFrame({'label': [0, 1]}).shape == (2, 1)
    assert np.allclose(scipy.linalg.solve(np.eye(2), [1., 2.]), [1., 2.])
    assert accuracy_score([0, 1], [0, 1]) == 1.0
    with tempfile.TemporaryDirectory(prefix='cgan-smoke-') as directory:
        target = Path(directory)
        checkpoint = target / 'generator.pt'
        torch.save(generator.state_dict(), checkpoint)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert all(torch.equal(value, restored[key]) for key, value in generator.state_dict().items())
        figure, axis = plt.subplots()
        axis.plot([0, 1], [float(disc_loss.detach()), float(gen_loss.detach())])
        figure.savefig(target / 'loss.png')
        plt.close(figure)
        assert (target / 'loss.png').stat().st_size > 100
        with SummaryWriter(str(target / 'events')) as writer:
            writer.add_scalar('smoke/generator_loss', float(gen_loss.detach()), 3)
        events = EventAccumulator(str(target / 'events')).Reload()
        assert events.Scalars('smoke/generator_loss')[0].step == 3
        notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(
            "import sys, torch\nassert '.cloud-environments/cgan/venv' in sys.executable\n"
            "x = torch.tensor([2.0], requires_grad=True)\n(x * x).sum().backward()\n"
            "assert x.grad.item() == 4.0\nprint('CGAN kernel OK')"
        )])
        NotebookClient(notebook, timeout=60, kernel_name='cgan-research', resources={'metadata': {'path': str(target)}}).execute()
        assert notebook.cells[0].execution_count == 1
        assert any('CGAN kernel OK' in output.get('text', '') for output in notebook.cells[0].outputs)
    print(json.dumps({'result': 'PASS', 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                      'device': 'cpu', 'synthetic_training_steps': 3,
                      'checks': ['generator and discriminator updates', 'finite gradients and losses',
                                 'checkpoint roundtrip', 'torchvision transforms and NMS',
                                 'NumPy/pandas/SciPy/scikit-learn', 'plot output',
                                 'TensorBoard event roundtrip', 'notebook kernel execution']}))


if __name__ == '__main__':
    main()
CGAN_SMOKE
uv pip check --python "$cgan_root/venv/bin/python"
cd /workspace/CGAN_Research
python "$cgan_root/smoke.py"
