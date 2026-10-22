# V2 research implementation plan

Status: prepared for review; implementation has not started.

**Goal:** Build a reproducible calendar-conditioned cGAN, corrected evaluation,
paper-style figures, and a Colab GPU workflow for my measured daily load data.

**Architecture:** A single Python package separates data ingestion, preparation,
models/training, metrics/evaluation, and figures. A command interface and a thin
Colab notebook use that same package. Experiments consume immutable split and
preprocessing artifacts and record every configuration, seed, and result.

**Stack:** PyTorch, NumPy, pandas, SciPy, Matplotlib, PyYAML, requests, and pytest.
Python 3.12 is available in the development environment; the reported Colab
traceback uses Python 3.13. Those are the initial runtime targets, with package
compatibility and execution recorded before either is claimed as supported.
Colab retains its compatible CUDA-enabled PyTorch.

**Specification:** [Approved research design](v2-design.md), including the
confirmed absence of EV labels. Optional inferred EV patterns are discussed in
[a separate note](ev-labeling.md) and are not part of these implementation tasks.

## Global constraints

- Work and pushes remain on `V2`; `main` is unchanged.
- No EV ownership, adoption-date, event, or charger-level labels enter the
  baseline. Unknown EV status is not a no-EV label. PV metadata is unverified.
- Baseline conditions are verified `season` and `day_type`, with up to eight
  observed classes. Timestamp/calendar requirements must be explicit.
- Daily profiles have exactly 24 finite hourly readings, with units and interval
  convention declared. Preserve legitimate negative net-load values.
- Proposed household split fractions are 70/15/15, split seed 42. Do not silently
  substitute random-row splits when household identifiers are unavailable.
- Fit normalization, feature scales, histogram bins, and condition mappings on
  training data only. Freeze test data from selection and tuning.
- G widths: `(24+K)→64→128→64→24`; D widths: `(24+K)→64→128→256→1`;
  LeakyReLU slope 0.2; D dropout 0.4; 24-dimensional standard-normal noise.
- Adam: G learning rate 0.0001, D learning rate 0.0002, betas (0.9, 0.999),
  epsilon 1e-8, weight decay 0; batch size 2,000; maximum 10,000 epochs.
- One D and one G update per batch; D loss averages real and fake losses.
  Logits plus BCEWithLogitsLoss replace the equivalent sigmoid-plus-BCE pair.
- Reference mode keeps input order fixed. Epoch shuffling and balanced sampling
  are separate declared ablations.
- Generate 2,000 samples per supported condition per seed. Planned seeds are
  17, 42, 2025, 31415, and 27182. Pilot results remain explicitly provisional.
- Corrected quantiles use linear 0.025 and 0.975 quantiles. Sample standard
  deviations use ddof=1. Continuous-feature KL uses 30 training-defined bins,
  underflow/overflow bins, and pseudocount 0.5; sensitivity uses 15/30/60 bins.
- Condition cells with fewer than two real samples have no variance-based
  metric; fewer than 50 test profiles are marked exploratory. Missing is not 0.
- Never claim paper-equivalent results, GPU validation, privacy guarantees, or
  scientific improvement from a synthetic integration fixture.
- Preserve upstream MIT attribution. Documentation uses first-person research
  or neutral project language, without conversational workflow commentary.
- Runtime defaults must be bounded and explicit. A full experiment is not
  launched implicitly by installing the package or opening the notebook.

## Review focus

1. Valid negative net load, constant profiles, and test extremes must not be
   silently discarded or clipped. Owned by Tasks 1, 2, and 4.
2. Missing household/date metadata, sparse conditions, and unknown EV status
   must block unsupported claims rather than trigger invented labels or splits.
   Owned by Tasks 1, 2, and 5.
3. Unequal sample counts and permuted rows must not change distribution metrics
   incorrectly or conceal undefined results. Owned by Task 4.
4. Interrupted runs must resume with compatible data, exact training state, and
   an intact last checkpoint; a lost Colab session must not lose all artifacts.
   Owned by Tasks 3 and 7.
5. Malformed assets, extraction paths, and figures for unavailable classes must
   fail clearly without overwriting files or presenting missing results as
   success. Owned by Tasks 1 and 6.

## Package and artifact boundaries

`src/cgan_research/` contains `data.py`, `download.py`, `prepare.py`, `models.py`,
`training.py`, `checkpoint.py`, `features.py`, `metrics.py`, `evaluation.py`,
`figures.py`, and `cli.py`. Shared dataclasses live in `types.py`. Public command
entry point: `cgan-research`. Configurations live in `configs/`; synthetic test
fixtures live in `tests/fixtures/`. No measured dataset is committed.

Prepared data and run outputs are supplied through explicit paths. Each run has
`config.json`, `provenance.json`, `status.json`, `history.csv`, checkpoint files,
generated samples, metric/figure source tables, and exported figures. A unique
run ID prevents accidental overwrite. Dataset, output, notebook checkpoint,
virtual-environment, and local credential paths are excluded from Git.

## Task 1: validated intake and calendar-label contract

**Files:** Create `pyproject.toml`, `.gitignore`, `src/cgan_research/__init__.py`,
`types.py`, `data.py`, `download.py`, `tests/test_data.py`,
`tests/test_download.py`, and a small synthetic CSV fixture.

**Interfaces:** `DatasetConfig` explicitly maps 24 hourly columns, profile ID,
household ID, date, timezone, units, interval convention, season month groups,
weekend days, and optional supplied holiday dates. `DailyProfiles` carries IDs,
optional household IDs, local dates, an `(N,24)` float64 array, condition fields,
units, and source fingerprint. `AuditReport` carries accepted/quarantined counts
and reason codes. `read_daily_csv(path, config) -> (DailyProfiles, AuditReport)`;
`download_asset(url, destination, expected_sha256=None) -> AssetManifest`.

- [ ] Write tests asserting: the canonical 24-hour fixture loads; a legitimate
  negative value survives; a nonfinite reading is quarantined with its reason;
  conflicting household/day duplicates are rejected; absent date/household
  fields remain explicit; a configured EV-related condition is rejected.
- [ ] Write tests asserting: declared month groups/weekend days give the
  expected calendar conditions; holidays affect day type only when supplied;
  missing dates cannot silently become calendar labels. A flat daily profile
  remains valid. Raw fields named PV or EV do not automatically become labels.
- [ ] Test timezone-aware daylight-saving transition dates, including a wide
  row that superficially has 24 columns on a local 23/25-hour day. Quarantine
  those rows under the initial protocol and count exclusions by month/season.
  Fixed-offset/UTC timestamps do not by themselves establish a local timezone.
- [ ] Test streamed HTTPS download to a temporary file, atomic final rename,
  SHA-256 mismatch, HTML-error responses, automatic source-archive URLs,
  archive traversal, duplicate member paths, and expanded-size limits. Do not
  fetch public data in unit tests; use controlled local HTTP response fixtures.
- [ ] Run `python -m pytest tests/test_data.py tests/test_download.py -q`; confirm
  failure from missing package functionality before implementing it.
- [ ] Implement the interfaces with explicit configuration validation. Initially
  accept mapped daily wide CSV. Do not guess a raw long-format adapter before
  the actual asset is inspected. Add ZIP support only with safe selected-member
  extraction and a configurable expanded-size ceiling; reject unsupported file
  types with an actionable error. Include the third-party license notice.
- [ ] Run the same tests and require all to pass. Commit the complete intake
  capability as `Add validated daily-load intake and calendar conditions`.

## Task 2: reproducible preparation without leakage

**Files:** Create `prepare.py`, `tests/test_prepare.py`, and
`configs/calendar_baseline.yaml`; extend `types.py`.

**Interfaces:** `prepare_dataset(data, config, output_dir) -> PreparedManifest`;
`load_prepared(path) -> PreparedData`. The manifest records ordered profile IDs,
household assignments, train/validation/test index arrays, condition mapping,
global training min/max, fingerprints, and provenance. Feature-specific scales
and histogram references are added by Task 4 after feature definitions exist.
The numerical store is a non-pickled NPZ file containing physical float64 loads
and model-space float32 loads, with an explicit representation tag and units.
`encode_conditions(frame, mapping) -> (integer_labels, supported_mask)` never
invents a held-out class. `transform_loads(physical, normalizer)` and
`inverse_loads(normalized, normalizer)` share the recorded training parameters.

- [ ] Write tests with at least 20 synthetic households that assert complete,
  disjoint household assignments and reproducibility for seed 42; verify a
  duplicate measurement cannot silently appear in multiple partitions.
- [ ] Assert that changing only held-out extreme loads does not change training
  min/max or fitted preprocessing. Preserve out-of-range transformed test
  values. For a constant training range, store scale 1 and a constant-range flag.
- [ ] Test transform/inverse round trips with a negative nonzero minimum,
  nonunit range, constant training loads, and held-out extremes. Assert correct
  dtype and representation metadata so no metric can silently mix physical
  and normalized values.
- [ ] Assert that absent household IDs blocks the default split with an explicit
  request for a revised protocol; missing dates blocks calendar conditioning;
  a condition seen only outside training is reported unsupported, not remapped.
- [ ] Run `python -m pytest tests/test_prepare.py -q` and confirm the new tests
  fail before implementation.
- [ ] Implement a stable household-list shuffle and recorded assignments, with
  documented integer rounding and minimum nonempty partitions. Fit all learned
  preprocessing on training only. Refuse output-directory overwrite. Store
  the full baseline constants from this plan in the baseline configuration.
- [ ] Run intake and preparation tests together. Commit as
  `Add frozen household splits and train-only preprocessing`.

## Task 3: reference cGAN and resumable training

**Files:** Create `models.py`, `training.py`, `checkpoint.py`,
`tests/test_models.py`, `tests/test_training.py`, and `tests/test_checkpoint.py`.

**Interfaces:** `Generator(num_conditions)` consumes float noise `(B,24)` and
integer conditions `(B,)`, returning `(B,24)`; `Discriminator(num_conditions)`
returns logits `(B,1)`. `train(prepared, config, run_dir, resume=None,
validation_callback=None) -> RunResult`; the optional callback consumes the
current generator and completed optimizer-step count and returns a
`ValidationResult(score, available, metadata)`. Task 5 implements the corrected
validation callback and selection integration. Early mechanical tests use no
callback and cannot produce a selected-best-model claim.
`generate(checkpoint, conditions, samples_per_condition, seed) -> GeneratedBatch`
inverse-transforms model outputs to physical float64 loads using checkpointed
normalization. The batch records units, physical representation, condition IDs,
seed, checkpoint fingerprint, and unsupported-condition errors.
`save_checkpoint(path, state)` is atomic; `load_checkpoint(path, expected) ->
TrainingState` validates fingerprints and architecture before mutation.

- [ ] Test exact layer widths, invalid/out-of-range labels, output dimensions,
  finite losses/gradients, and actual updates of both networks on a synthetic
  two-condition fixture. Assert that D-only updates do not update G parameters.
- [ ] Test generated-output inverse transformation with nonzero training minima
  and nonunit ranges. Saved profiles must be in the same physical units as
  real data; output ranges must not be clipped merely to improve metrics.
- [ ] Test one four-step CPU run against two steps plus checkpoint/resume plus
  two steps: model/optimizer state, batch order, noise draws, step count, and
  subsequent results must match in the same environment. Test mismatched data,
  changed configuration, partial checkpoint files, and preservation of the
  last valid checkpoint after an interrupted write.
- [ ] Run `python -m pytest tests/test_models.py tests/test_training.py
  tests/test_checkpoint.py -q` and confirm the new capability is missing.
- [ ] Implement the exact baseline constants, per-step and per-epoch histories,
  declared fixed/shuffled/balanced sampling modes, a complete batch cursor and
  sampling state for resume, RNG state, and atomic checkpoints. Store only
  tensors and primitive metadata compatible with `weights_only=True`; do not
  deserialize arbitrary downloaded Python objects. Use a distinct generation
  RNG so validation sampling does not perturb subsequent training randomness.
- [ ] Run model/training/checkpoint tests plus preparation tests. Commit as
  `Implement reference cGAN with deterministic checkpoint recovery`.

## Task 4: corrected features and distribution metrics

**Files:** Create `features.py`, `metrics.py`, `tests/test_features.py`, and
`tests/test_metrics.py`.

**Interfaces:** `daily_features(loads, mode='corrected') -> FeatureTable` with
near-peak, near-base, peak-duration, rise-time, and fall-time columns;
`fit_metric_reference(train_features, bins=30) -> MetricReference`;
`compare_distributions(real, generated, reference) -> MetricResult`. Return
numeric values plus availability and reason codes, never substitute missing 0.
Feature extraction requires physical-unit inputs. Add
`fit_reference_from_prepared(prepared, metric_config, output_path)` to persist
training-only feature scales, constant-feature flags, and bin edges alongside
the prepared-data fingerprint. Task 6's preparation command calls this stage
after Task 2; earlier preparation tests do not require the metrics module.

- [ ] Pin quantile behavior: for `np.arange(24)`, base=0.575 and peak=22.425.
  Test constant, tied-peak, and midnight-crossing profiles against the design's
  cyclic rise/fall definition. Compatibility order statistics must instead
  return base=1 and peak=22 on that fixture and be labeled separately.
- [ ] Test KL/JS identity, known discrete probability examples, permutation
  invariance, unequal sample counts, empty classes, overflow, pseudocount
  normalization, and a constant training feature. Test scalar Gaussian Fréchet
  against the analytic mean/standard-deviation formula with ddof=1.
- [ ] Test joint Fréchet identity, known diagonal covariance cases, singular
  positive-semidefinite covariance, and invalid/nonfinite input. Significant
  negative eigenvalues must fail rather than be silently repaired. Test that
  changing held-out values never refits training histogram edges or scales.
- [ ] Run `python -m pytest tests/test_features.py tests/test_metrics.py -q` and
  confirm missing functionality before implementation.
- [ ] Implement fixed integer bins for time features, training-defined continuous
  bins with explicit tails and smoothing, and a symmetric PSD square-root
  formulation for joint Fréchet. Bound only numerical roundoff under a documented
  tolerance; preserve diagnostic flags. Add Wasserstein and Jensen–Shannon
  alongside KL. Expose upstream calculations only under a compatibility label.
- [ ] Pass all metric tests, including the row-permutation counterexample from
  the source audit. Commit as `Add corrected load-profile metrics and invariance tests`.

## Task 5: experiment evaluation and comparison records

**Files:** Create `evaluation.py`, `tests/test_evaluation.py`,
`tests/test_experiment_records.py`, and `configs/calendar_ablations.yaml`.

**Interfaces:** `evaluate(prepared, generated, reference, split) -> Evaluation`;
`validation_score(evaluation) -> SelectionScore`;
`compare_runs(run_dirs, bootstrap_seed, resamples) -> ComparisonReport`.
Tables include condition, feature/metric name, split, model, seed, real/generated
counts, units, metric version, value, availability reason, and exploratory flag.
Evaluation rejects normalized input or mismatched units/fingerprints. It reads
the physical arrays produced by Task 2 and Task 3.

- [ ] Test equal-class macro and real-count-weighted summaries on deliberately
  imbalanced fixtures. Mark unsupported/<2-sample variance cells unavailable,
  <50-sample cells exploratory, and include excluded counts in every summary.
- [ ] Test that checkpoint selection reads validation only and uses macro
  Wasserstein scaled by training feature standard deviation; exclude and report
  training-constant features from that selection score. If no feature remains,
  return unavailable and block score-based selection rather than returning 0.
- [ ] Test that real and generated row permutations do not alter profile-level
  mean RMSE or distribution metrics. Test all sample counts and seeded outputs
  remain reproducible. Test duplicate/memorization diagnostics without exposing
  household identifiers in a public report.
- [ ] Test physical-unit comparisons using real/generated profiles with a
  negative minimum and nonunit range; identical physical distributions score
  zero where mathematically expected, and a normalized batch is rejected.
- [ ] Define daily energy as interval-energy summation, or integration over
  declared one-hour mean-power intervals. Define within-day ramps as the 23
  adjacent-hour differences without an invented midnight boundary. Report their
  Wasserstein distances; test constant and linear-ramp analytical fixtures.
- [ ] Report mean within-profile autocorrelation at lags 1–6, with constant
  profiles explicitly unavailable, plus mean pairwise RMS distance estimated
  from 1,000 distinct-index profile pairs sampled with seed 42 as a diversity
  diagnostic. Pin the centered autocorrelation denominator to the full-day sum
  of squared deviations. Test constant, repeated, and varied-profile fixtures.
- [ ] Implement real-versus-real finite-sample references using two independent
  with-replacement resamples from the selected real partition, each of the same
  size as the corresponding generated comparison, with 100 replicates and seed
  42. Label this a sampling reference, not a performance bound. Verify that the
  separate empirical generation baseline draws only from training records.
- [ ] Execute and save continuous-feature KL sensitivity tables for 15/30/60
  training-fitted bins. Tests must confirm that all settings are reported and
  none is selected retrospectively for the best test score.
- [ ] Test paired seed summaries and household-cluster resampling using small
  known fixtures; no row-wise independence assumption may replace household
  uncertainty. Define 1,000 bootstrap replicates and bootstrap seed 42 for the
  report configuration, retaining explicit limitations for very few households.
- [ ] Run the new tests, implement the interfaces, and require all to pass.
  Persist failed run statuses as well as successful runs. Implement the empirical
  within-condition resampling baseline and declared training-order ablations.
- [ ] Integrate scheduled validation into `training.py` and `checkpoint.py`.
  Full-run configuration must provide a positive validation interval in optimizer
  steps and a fixed evaluation-noise seed; no automatic interval is inferred
  from test scores. Select the lowest available validation score, keeping the
  earliest step on ties. Store best score, best step, next validation step,
  evaluation RNG state, and selected-checkpoint path in resumable state. Reuse
  fixed per-condition validation noise without advancing training RNG state.
- [ ] Extend the uninterrupted/resume test across a validation boundary and
  assert identical history, selection state, and best checkpoint. Verify that
  validation uses validation records only and that sampling ablations receive
  equal maximum optimizer-step budgets, not merely the same epoch count.
  Commit as `Add reproducible model comparisons and evaluation records`.

## Task 6: figures and command-line workflow

**Files:** Create `figures.py`, `cli.py`, `__main__.py`,
`tests/test_figures.py`, `tests/test_cli.py`, and `tests/test_integration.py`;
register the package's console entry point.

**Interfaces:** `render_figures(evaluation_dir, output_dir, figure_config) ->
FigureManifest`. Commands: `download`, `audit`, `prepare`, `train`, `generate`,
`evaluate`, and `plot`, each with explicit input/config/output arguments.
Commands preserve exit codes and return nonzero on failed prerequisites.

- [ ] Test correct real/generated series and saved source tables, accurate units,
  fixed comparative scales, overflow warnings, and unavailable-class markers.
  The no-EV-label figures must contain no invented EV scenarios. Test PDF/SVG
  and 300-DPI PNG output and successful headless rendering.
- [ ] Test commands from a different working directory, paths containing spaces,
  an existing run directory, malformed configuration, and unsupported dataset
  type. A failed audit cannot continue silently into training.
- [ ] Run an end-to-end synthetic integration fixture through preparation,
  two short training epochs, generation, evaluation, and plotting. Assert
  expected table rows and artifacts, not just process exit status. The run
  manifest and figures must identify the fixture as synthetic.
- [ ] Implement CLI and figures around the existing interfaces. Reproduce the
  paper's plot families with adapted calendar facets: weekday/weekend seasonal
  mean profiles, mean/std heatmaps, declared peak/base distribution panels,
  KL heatmaps, and feature Fréchet heatmaps. Add original workflow and model
  diagrams; do not copy or trace the paper's curves.
- [ ] Run the complete test suite and visually inspect representative figure
  exports against their saved tables. Commit as
  `Add reproducible paper-style figures and research commands`.

## Task 7: Colab integration and measured-data pilot

**Files:** Create `notebooks/V2_Colab.ipynb`, `docs/colab.md`,
`tests/test_notebook.py`, and `docs/research/experiment-register.md`; update
README usage and status. Add a dataset-specific configuration only after the
actual asset has been inspected.

**Interfaces:** The notebook invokes the packaged CLI and saves its run directory
to an explicitly selected persistent Drive location. It records the exact V2
commit, Python/PyTorch/CUDA/GPU versions, available memory, data fingerprint,
configuration, and command outcomes. It supports an explicit resume path.

- [ ] Test notebook validity and that it uses the package instead of a duplicate
  training implementation. Assert that neither the CPU installer nor a CPU-only
  torch index is invoked. Execute a CPU synthetic notebook smoke path separately
  from the GPU research path and mark its output accordingly.
- [ ] Implement notebook cells for branch checkout, environment/GPU inspection,
  compatible dependency installation, optional authenticated Drive mounting,
  dataset download/audit/preparation, bounded pilot, train/resume, evaluation,
  plotting, and artifact persistence. Do not start a full run automatically.
- [ ] Once the actual dataset is attached, inspect it, supply its explicit
  column/unit/calendar mapping, and run the complete audit. A long-format or
  non-CSV input requires a small adapter plan and tests grounded in the actual
  schema. Do not replace missing real data with upstream generated samples.
- [ ] Run a measured-data pilot, record GPU throughput and memory, and freeze
  a concrete GPU-time budget, evaluation interval, and candidate configurations
  in the experiment register before launching full runs. These values depend
  on measured throughput and must not be invented before the pilot.
- [ ] Run all declared seeds and candidates within that budget; retain failed
  attempts. Select on validation, then evaluate the frozen final test once.
  Report results and limitations even if the candidate is worse. A budget or
  session limit is an explicit continuation point, not a claim of completion.
- [ ] Validate persisted artifacts after a Colab restart and confirm resume.
  Commit the notebook, adapter/configuration, and documentation on `V2`, keeping
  large data, checkpoints, and private records out of Git.

## Delivery and outstanding inputs

Tasks 1–6 can be implemented and checked with clearly marked synthetic fixtures
once this plan is reviewed. The actual data schema must still be inspected
before claiming that an importer supports my dataset. Task 7's measured-data
and GPU checks require the real asset and execution in my Colab runtime.

As rechecked on 22 October 2026, the release has no measured-data attachment.
The lack of EV labels is already resolved as a baseline design constraint; no
EV record is required to start the calendar-conditioned project. Independent EV
records, if they become available, are only a prerequisite for a separate
validated EV-labeling experiment.

Before each push, run the tests appropriate to the changes, check staged files
for unintended data or credentials, and verify the destination is `V2`. Use the
configured repository author identity and ordinary descriptive commit messages.
Do not rewrite existing published history as part of this plan.
