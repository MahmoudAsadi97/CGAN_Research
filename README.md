# CGAN Research

I am adapting the conditional generative adversarial network in
[Li, Dong, and Qiu (2025)](https://doi.org/10.1016/j.enbuild.2025.115584) to my
electricity-load dataset. My goal is to generate realistic 24-hour load profiles,
reproduce the paper's evaluation figure types, and assess improvements through
controlled experiments on held-out data.

## Research scope

My dataset includes provider-supplied PV, heat-pump, and **EV-detection**
indicators. The EV indicator is not verified ownership or a charging-event
label: zero means no EV was detected, not proven absence. The approved initial
model uses season and calendar day type. These newly inspected indicators can
support a documented extension, with their source definitions preserved.

I may investigate inferred charging-related patterns as a separate experiment.
Without charger records or another independent reference, those patterns cannot
be described as verified EV ownership or charging events.

I will compare the reference cGAN architecture and candidate improvements using
the same splits, preprocessing, metrics, sample counts, and seeds. The reference
implementation has evaluation discrepancies documented in the source audit,
so I keep corrected metrics separate from historical compatibility diagnostics.

## Current status

I have implemented the electricity-only calendar cGAN pipeline on `V2`: audited
intake, frozen meter-group splits, training-only preprocessing, bounded resumable
training, corrected evaluation, comparison records, figures, and a thin Colab
notebook. I use the existing `gpu_env`; I do not replace its PyTorch installation.

I prepared 873,600 daily profiles from all eight local electricity cohorts after
excluding 4,800 DST meter-days. My local archive differs from the previously
audited release checksum, so I record it as an **unverified local variant**.
Matching observed coverage does not establish byte equivalence. Meter IDs are
household proxies, not independently verified household identities.

My current managed session blocks GPU access, Git metadata writes, and GitHub
DNS access. CPU tests and measured-data intake checks do not establish a GPU
pilot result. I record actual commands, results, review findings, and remaining
limitations in the [experiment register](docs/research/experiment-register.md).
I have not evaluated model performance on the frozen test set or established an
improvement or publication readiness. `main` remains unchanged.

## Local workflow

I run the commands from the repository root with my existing environment:

```bash
conda activate gpu_env
export PYTHONPATH="$PWD/src"
export MPLCONFIGDIR="$PWD/outputs/matplotlib"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q

python -m cgan_research prepare --kind electricity \
  --input /path/to/raw.zip --config configs/dataset_electricity.yaml \
  --output outputs/prepared

python -m cgan_research train --prepared outputs/prepared \
  --config configs/calendar_baseline.yaml --output outputs/pilot-001

python -m cgan_research generate --checkpoint outputs/pilot-001/best.pt \
  --samples-per-condition 2000 --seed 42 --output outputs/pilot-001/generated.npz

python -m cgan_research evaluate --prepared outputs/prepared \
  --generated outputs/pilot-001/generated.npz --split validation \
  --output outputs/pilot-001/validation

python -m cgan_research plot --evaluation outputs/pilot-001/validation \
  --output outputs/pilot-001/figures
```

I use a new output path for each preparation, run, evaluation, and figure export.
I resume in the original run directory by adding `--resume outputs/pilot-001/last.pt`
to the same training command and retaining the exact configuration. The pilot is
bounded by 400 optimizer steps and 600 cumulative seconds, with validation every
100 steps and checkpoints every 25 steps. Time checks occur between batches;
validation and checkpoint writes can overrun the time limit. The default device
is explicitly CUDA and never falls back to CPU. If no validation checkpoint has
been selected, `best.pt` is unavailable; `last.pt` is a recovery checkpoint.

I can generate the empirical within-condition training-resampling baseline with
`generate --empirical --prepared outputs/prepared --seed 42 --experiment-seed 17`
for a comparison paired with training seed 17 and generation seed 42. I compare saved evaluations
using `compare --evaluations <reference-evaluation> <candidate-evaluation> ...
--output <comparison.json>`. Paired comparisons require shared seeds, split,
metric references, units, and generated sample counts. Whole-meter cluster
intervals use the private grouped records saved separately from public tables.
Neither empirical resampling nor nearest-profile diagnostics establish privacy.

I keep datasets, private prepared manifests, checkpoints, samples, and figures
in ignored `outputs/` directories or another external path. Public evaluation
JSON/tables omit meter identifiers; `private_records.npz` remains private. My
calendar uses Belgian local time, northern-hemisphere seasons, Saturday/Sunday
weekends, and an explicitly empty holiday list. I do not claim holiday adjustment.
Temperature and provider technology flags do not enter baseline conditions.

## Documentation

- [Research design](docs/research/v2-design.md)
- [Implementation plan](docs/research/implementation-plan.md)
- [Paper and source-code audit](docs/research/source-audit.md)
- [Source provenance](docs/research/source-manifest.json)
- [Dataset availability and quality audit](docs/research/dataset-audit.md)
- [EV-labeling considerations](docs/research/ev-labeling.md)
- [CPU development environment](docs/cloud-environment.md)
- [Laptop GPU setup and readiness check](docs/local-gpu.md)
- [Experiment register and verification](docs/research/experiment-register.md)
- [Portable Colab workflow](docs/colab.md)

The existing `scripts/setup-cloud.sh` is for the CPU cloud workspace. I do not
run it over Colab's GPU-enabled PyTorch installation. My Colab notebook
runs the same research package and preserve logs, checkpoints, evaluation
tables, and figures between sessions.

## Reference and attribution

The reference code is [malvin0130/cGAN_EB](https://github.com/malvin0130/cGAN_EB),
inspected at commit `885f78052a91fbc755e6b17a354c68849b5556ca`. Its
[MIT license](docs/research/upstream-MIT-LICENSE.txt) is retained. The original
measured dataset is confidential; the public reference CSVs are generated
samples and are not my training data.

> Li, Y., Dong, B., & Qiu, Y. (2025). Conditional generative adversarial network
> (cGAN) for generating building load profiles with photovoltaics and electric
> vehicles. *Energy and Buildings*, 335, 115584.
> https://doi.org/10.1016/j.enbuild.2025.115584
