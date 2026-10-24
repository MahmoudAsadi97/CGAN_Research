"""Execute the independent CPU notebook path against the real package CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'V2_Colab.ipynb'


def test_colab_notebook_valid_and_cpu_smoke_executes_package(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    smoke = [cell for cell in notebook.cells if 'cpu-smoke' in cell.metadata.get('tags', [])]
    assert len(smoke) == 1
    repo = NOTEBOOK.parents[1]
    env = dict(os.environ, CGAN_REPO=str(repo), CGAN_SMOKE_DIR=str(tmp_path / 'cpu synthetic smoke'),
               PYTHONPATH=str(repo / 'src'), MPLBACKEND='Agg', OPENBLAS_NUM_THREADS='1')
    executed = subprocess.run([sys.executable, '-c', 'RUN_CPU_SMOKE = True\n' + smoke[0].source], env=env, cwd=tmp_path,
                              capture_output=True, text=True, timeout=120)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    output = tmp_path / 'cpu synthetic smoke'
    prepared = json.loads((output / 'prepared' / 'manifest.json').read_text())
    assert prepared['synthetic'] and prepared['counts']['train']['groups'] == 28
    status = json.loads((output / 'run' / 'status.json').read_text())
    assert status['steps'] == 2
    evaluation = json.loads((output / 'evaluation' / 'evaluation.json').read_text())
    assert evaluation['metadata']['split'] == 'validation'
    assert evaluation['metadata']['provenance']['synthetic'] is True
    assert evaluation['metadata']['n_generated_total'] == 8
    assert len(evaluation['rows']) == 80
    figures = json.loads((output / 'figures' / 'figure_manifest.json').read_text())
    assert figures['synthetic'] and len(figures['figures']) == 8
    assert (output / 'figures' / 'mean_profiles.svg').is_file()


def test_research_cells_are_inert_until_explicit_opt_in(tmp_path):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    research = [cell for cell in notebook.cells if 'research-action' in cell.metadata.get('tags', [])]
    assert len(research) >= 6
    # Default cells execute without Colab, network access, data, or training.
    script = '\n\n'.join(cell.source for cell in research)
    executed = subprocess.run([sys.executable, '-c', script], cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert list(tmp_path.iterdir()) == []
