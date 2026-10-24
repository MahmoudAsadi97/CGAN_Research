# Colab research workflow

I use [V2_Colab.ipynb](../notebooks/V2_Colab.ipynb) as a thin interface to the
same `python -m cgan_research` commands used locally. Research actions default
to disabled. I enable each action after inspecting its inputs and outputs.
This notebook has a locally executed CPU synthetic smoke path. A real Colab
GPU run, Drive restart recovery, and measured-data pilot remain unverified.
The tested local environment is Python 3.10.12, PyTorch 2.5.1 built for CUDA
12.1; CPU execution does not establish GPU readiness. The notebook records the
actual runtime versions rather than assuming Colab matches this environment.

I select a GPU runtime, enable V2 checkout, inspect its SHA, and optionally set
`EXPECTED_COMMIT` to pin a reviewed commit. Checkout refuses local changes.
Installation uses `pip install --no-deps -e` and installs the project’s other
dependencies separately, retaining Colab’s existing Torch installation. I do
not run a CPU cloud installer. `PYTHONPATH=src` is also supported when an
editable package installation is unavailable.

I explicitly enable Drive mounting and authenticate in Colab, then set
`WORK_ROOT` to my chosen persistent folder. All prepared data, checkpoints,
command logs, metrics, and figures live below that path. Spaces in paths are
supported. An explicitly chosen `/content/...` path is ephemeral. I inspect a
saved checkpoint after restarting the runtime before claiming persistence.
I keep raw data, household/meter identifiers, and checkpoints outside Git.

I enable `RUN_DOWNLOAD` to retrieve the release declared in
`configs/dataset_electricity.yaml`, including SHA-256 verification. I enable
`RUN_PREPARE` to audit the electricity archive before preparing immutable
meter-group splits and training-only preprocessing. The adapter sums
quarter-hour net kWh, preserves negative exchange, and reports incomplete and
DST-day exclusions. Its meter grouping is a household proxy. Weather,
provider EV-detection flags, PV, and heat-pump flags remain outside this
calendar baseline. Archive size and traversal validation remain in the package.

I enable either `RUN_PILOT` or `RUN_RESUME`. The pilot is bounded by 400 steps,
600 cumulative seconds checked between batches, and 10,000 epochs, stopping
at the first limit. Resume requires the original run path, explicit checkpoint,
data fingerprints, and identical configuration; it does not increase the
original budget. `status.json` records actual steps, elapsed time, throughput,
and peak allocated GPU memory. `environment.json`, the run’s provenance, and
`commands.jsonl` retain versions, commit information, configurations, and command
outcomes. These artifacts support a later measured budget decision; I do not
infer a full experiment budget from CPU smoke timing.

`RUN_EVALUATE` generates every sample and evaluates validation using the selected
best checkpoint when available. A last-checkpoint fallback is provisional.
Opening the frozen test partition is a separate deliberate CLI action requiring
`--confirm-test`. `RUN_PLOT` exports all calendar panels, including unavailable
classes. Output directories are immutable; I use a new path for a new evaluation
rather than overwriting evidence.

For a standalone local CPU smoke, I set `CGAN_REPO` to the checkout and
`CGAN_SMOKE_DIR` to a new output directory, then execute only the notebook cell
tagged `cpu-smoke` with `RUN_CPU_SMOKE = True`. The cell creates 40 synthetic
households, audits/prepares their profiles, performs two CPU optimizer steps,
generates eight profiles, evaluates validation, and renders the figures through
the real package CLI. Its manifests and figures identify synthetic data.
No download, GPU, or Drive authentication is required. This integration check
supports software behavior only; it is not a research performance result.

## Figure evidence and comparison scales

Each plot output includes eight families: original workflow and model diagrams,
seasonal mean profiles, mean/std difference heatmaps, KL heatmaps, per-feature
Gaussian Fréchet heatmaps, and near-peak/near-base histograms. All are exported
as PDF, SVG, and 300-DPI PNG. Real profiles are orange and generated profiles
are teal. Calendar panels use all meter groups and contain no invented EV
scenarios. Missing classes are gray and marked unavailable. Temporal and
load-valued features have separate colorbars with their own units.

`source_tables/evaluation.json` preserves the exact evaluation record and all
list-valued tables are also saved as CSV. `figure_manifest.json` (also saved as `manifest.json` for CLI consumers) records the
split, model, seed, estimator version, units, per-condition counts, synthetic
status, actual display limits, and saturation/tail warnings. Histograms show
raw empirical bin fractions with fixed training edges. Unbounded tails are
counted separately in annotations and the manifest, and remain in the source
tables. Missing values are never encoded as zero.

I select one fixed figure YAML **before comparing models** and use it for every
candidate. The following values illustrate configuration syntax only; I choose
and freeze appropriate ranges using training/validation evidence, not test
results:

```yaml
profile_ylim: [-2, 8]
heatmap_limits:
  mean_difference:
    near_peak: [-2, 2]
    near_base: [-2, 2]
    peak_duration: [-4, 4]
    rise_time: [-4, 4]
    fall_time: [-4, 4]
  std_difference:
    near_peak: [-2, 2]
    near_base: [-2, 2]
    peak_duration: [-4, 4]
    rise_time: [-4, 4]
    fall_time: [-4, 4]
  kl: [0, 2]
  frechet:
    near_peak: [0, 8]
    near_base: [0, 8]
    peak_duration: [0, 16]
    rise_time: [0, 16]
    fall_time: [0, 16]
histogram_xlim:
  near_peak: [-2, 8]
  near_base: [-2, 8]
```

I pass that path as `FIGURE_CONFIG` or `plot --config`. Limits omitted from the
YAML are evaluation-derived and recorded; those exports alone do not establish
fixed-scale model comparisons. The manifest and runtime warnings disclose
values beyond displayed ranges rather than silently dropping them.
