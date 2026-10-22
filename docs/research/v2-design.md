# V2 conditional load-profile research design

Status: baseline design approved on 2026-10-22, with EV ownership and charger
labels excluded. Implementation planning is in progress.

## Goal and constraints

Implement the method in Li, Dong, and Qiu (2025) and `malvin0130/cGAN_EB` on
my measured dataset, with a reproducible Colab GPU workflow and figures
corresponding to the paper. Establish whether controlled improvements outperform
the reference baseline under a credible held-out evaluation.

I keep all research code, documentation, and commits on `V2` in
`MahmoudAsadi97/CGAN_Research`. I will decide when to merge the work. The existing
checkout is already isolated; no additional worktree is needed. `main` remains
unchanged.

Evidence and source-specific discrepancies are recorded in
[the source audit](source-audit.md) and
[the provenance manifest](source-manifest.json).

No result, condition label, unit, dataset size, or privacy guarantee may be
invented. Synthetic demonstrations are marked as demonstrations. My
publication objective requires measured results; this specification is not a
claim of publication readiness or guaranteed improvement.

## Approach selection

1. **Recommended: audited reference baseline, then controlled improvements.**
   Preserve the reference network as a named baseline; implement reliable data
   handling, corrected metrics, experiment records, and the paper's figure
   families. Improve training only after the baseline is measured. This makes
   each gain attributable and supports a defensible methods section.
2. **Literal upstream script reuse.** This is quicker, but inherits missing
   real data, fixed labels, training-set evaluation, and metric discrepancies.
   Useful only for clearly labeled compatibility diagnostics.
3. **Replace the model immediately.** A more complex generative architecture
   might improve quality, but prevents isolating the contribution and adds
   tuning cost before data quality or the baseline is understood. Defer it.

## 1. Data and provenance contract

The release now contains `raw.zip`, downloaded and verified against the published
SHA-256. Its full contents and schema are recorded in the [dataset audit](dataset-audit.md).
Electricity readings are quarter-hour imported/exported energy in kWh, documented
in Belgian CET/CEST local time despite a literal `Z` suffix. Weather timezone,
units, and measurement provenance still require confirmation before joining it.
Retain the asset URL, byte count, local SHA-256, and publisher checksum. Source
integrity does not by itself establish measurement accuracy.

The internal daily representation will have:

- `profile_id`: a stable unique identifier for a daily profile;
- `household_id`: a stable pseudonymous grouping key when available;
- `date`: the local calendar day, with timezone documented at ingestion;
- `load_00` through `load_23`: finite numerical hourly readings in a declared
  physical unit and with interval start/end convention recorded;
- explicit condition columns justified by the dataset, plus a reversible
  `condition_id` mapping stored with the experiment.

Raw column names are mapped through an explicit dataset configuration, never
guessed from position. If input is long-format or subhourly, create a specific
adapter after inspection. Aggregate interval energy by summation; integrate
power using interval duration. Never use a generic averaging rule for both.

Audit duplicate household/day keys, repeated profiles, missing hours, irregular
sampling, nonfinite readings, unit changes, gaps, household coverage, and counts
for each proposed condition. Preserve legitimate negative net-load readings.
Quarantine incomplete days and ambiguous 23/25-hour daylight-saving days in the
initial protocol, recording counts and potential seasonal bias. Do not silently
impute, clip values, or infer EV/PV ownership from a load shape.

Season definitions and holiday jurisdiction must be explicit. The original
paper uses northern-hemisphere month groupings and treats holidays as weekends;
that convention is not assumed suitable for an unidentified new location.

My dataset has provider EV-detection flags, but no verified ownership or event
labels. The approved primary baseline excludes
EV ownership, EV adoption dates, charging events, and charger-level conditions.
It must not infer these attributes from electricity consumption or treat missing
EV information as a confirmed no-EV status.

The initial conditioning scheme is season and calendar day type, derived from
verified dates and an explicit calendar configuration. This gives up to eight
observed classes when all four seasons and both day types are represented. PV
status is supplied in the release and can support a separately documented
extension. The newly inspected heat-pump and EV-detection indicators also require
their source definitions to be preserved. Calendar configuration must be explicit
before deriving calendar labels.

Any later EV-labeling study is a separate, explicitly labeled experiment.
Verified ownership/adoption records describe household status; charger/submeter
records can identify charging events. Inferences from aggregate load alone are
pseudo-labels with uncertainty, not measured ownership or charging ground truth.
Synthetic charging injection establishes synthetic labels only. Without
independent validation, the project cannot make real-world EV detection or
EV-specific load-generation accuracy claims. The no-EV-label baseline remains
the primary comparison unless the research protocol is explicitly revised.

## 2. Splits and preprocessing

The primary claim to evaluate is generalization to held-out households. When
household identifiers exist, use disjoint household groups with approximately
70% training, 15% validation, and 15% test households and fixed split seed 42.
Store exact assignment files before training; assess condition coverage without
changing splits based on model scores. Report actual profile and household
counts rather than implying that household fractions equal profile fractions.

If identifiers do not exist but dates do, a chronological split is a different
generalization claim and needs a documented protocol revision after inspection.
If neither is available, an independent-household or temporal claim is not
supported. Do not silently fall back to random rows. Duplicate measurements
must not cross partitions; retain an audit of legitimate repeated values versus
duplicate records rather than deleting all flat or repeated days automatically.

Fit normalization, optional feature scaling, histogram bins, and any selection
statistics on training data only. The primary baseline uses a single global
training-set min–max transform, matching the reference code's transform family.
Record extrema and handle constant training data explicitly. Preserve test
values outside the training range and log them; do not clip them into range.
The paper's fixed-maximum transform may be exposed as a separately named mode.

The frozen test set is not used to select epochs, hyperparameters, samples,
metrics, or figure examples. If test results motivate another model-development
round, document that the test has become development data and reserve a new
independent confirmatory evaluation before making a confirmatory claim.

## 3. Baseline model and training

Use the released architecture with a configurable number K of observed classes:

- Generator: concatenate 24 noise values with K one-hot labels; dense widths
  64, 128, 64, and 24; LeakyReLU slope 0.2 between hidden layers; linear output.
- Discriminator: concatenate 24 hourly values with K one-hot labels; dense
  widths 64, 128, 256, and 1; LeakyReLU slope 0.2; dropout 0.4 before output.
- Binary adversarial losses. For the reliable implementation, use discriminator
  logits and `BCEWithLogitsLoss`, documenting the numerically stable equivalent
  of the upstream sigmoid-plus-BCE formulation. Average real/fake discriminator
  losses as the released code does.
- Adam learning rates G=0.0001 and D=0.0002; record upstream defaults
  betas=(0.9, 0.999), epsilon=1e-8, and weight decay=0.
- Reference configuration: batch size 2,000, maximum 10,000 epochs, one D update
  and one G update per batch, normal noise, 2,000 generated profiles per class.
  Preserve any smaller final batch and record actual optimizer steps.

The implementation detaches fake profiles during the discriminator update,
freezes discriminator parameters during the generator update, logs per-epoch
averages, and uses `eval()` / `no_grad()` for generation. Record these engineering
changes. A named reference-order mode preserves fixed input order; epoch
shuffling is a named ablation rather than an unreported baseline change.

I use CPU for unit and short integration checks. Full research runs target Colab's GPU.
Check actual GPU name, memory, CUDA/PyTorch versions, and throughput before
choosing a run budget. Do not automatically launch 10,000-epoch multi-seed runs
on this CPU workspace. Record shortened runs as pilots, never as full-budget
paper replications.

Checkpoints include both models, optimizers, completed step/epoch, Python/NumPy/
PyTorch RNG states, CUDA RNG states where available, configuration, normalization,
condition mapping, split fingerprint, and data fingerprint. Resume must reject
incompatible data/configuration. Save atomically and preserve the last valid
checkpoint on interruption. Reproducibility is scoped to the recorded software
and hardware environment; do not promise bitwise equality across devices.

## 4. Evaluation definitions

Store separate namespaces for `corrected` metrics and optional
`upstream_compatibility` diagnostics. The upstream KL-like expression must not
appear as a valid primary KL score or govern checkpoint selection.

For corrected daily features, use linear 2.5th and 97.5th quantiles. Peak duration
is the count of hourly values strictly above the midpoint of those quantiles,
in hours. For rise/fall, use an explicit cyclic-day definition: select the first
occurrence of the global maximum; walk backward/forward on the 24-hour cycle to
the nearest sample at or below the near-base threshold. The distances to that
peak are rise/fall hours. Flat profiles return zero times. This is a declared
operational definition, not a claim that the paper uniquely specifies that
algorithm. Also retain the reference code's index-based definition for clearly
labeled compatibility tables.

Primary distribution and profile diagnostics:

- Signed differences in each feature's mean and sample standard deviation
  (`ddof=1`); also record absolute differences for summaries.
- KL(real || generated) from shared probability histograms, never sample-value
  arrays. For load-valued features use 30 training-defined bins plus explicit
  underflow/overflow bins. Add pseudocount 0.5 per bin to both counts and
  normalize. For discrete hour-count features use fixed integer-centered bins
  covering 0–24 plus overflow handling. Record bins, smoothing, sample counts,
  and a predefined 15/30/60-bin sensitivity analysis for continuous features.
- Jensen–Shannon divergence and one-dimensional Wasserstein distance as
  complementary distribution metrics. State natural-log units for KL/JS.
- Per-feature Gaussian Fréchet distance, analytically `(mu_r-mu_g)^2 +
  (sd_r-sd_g)^2`, plus a separate joint-feature Fréchet distance using an
  explicitly train-fitted feature scaler. These are not Inception-image FID.
- Hourly mean-profile RMSE, daily-energy and ramp distributions, autocorrelation,
  and diversity diagnostics. Do not use row-paired MAE/RMSE between arbitrary
  independent generated and real days as a reconstruction score.
- Nearest-training-profile distances and exact-match rates as memorization
  diagnostics. These do not establish differential privacy or anonymization.

Define and test behavior for constant data, ties, missing classes, insufficient
samples, unequal real/generated counts, and covariance degeneracy. Fewer than
two real profiles makes variance-based metrics unavailable. Mark cells with
fewer than 50 test profiles as exploratory. Report all excluded/unavailable
conditions and sample counts; missing results must never be represented as zero.

Generate 2,000 samples per supported class per seed. Preserve every generated
sample for metrics; do not select visually attractive samples or remove poor
samples. Report equal-class macro summaries and real-count-weighted summaries
separately. Compare real-versus-real resamples as a finite-sample reference.

## 5. Experiment and improvement protocol

Start with the reference architecture and an empirical within-condition
training-data resampling baseline. The latter tests whether the learned model
adds value beyond copying the training distribution; it is not a privacy-safe
generator.

The planned training seeds are 17, 42, 2025, 31415, and 27182, sharing one frozen
data split. A pilot may use one seed to validate execution and estimate cost;
that is not a five-seed performance result. Use validation macro feature
Wasserstein distance, scaled by training feature standard deviations, for
checkpoint selection. Exclude only features constant in training from this
selection score, report the exclusion, and retain their error diagnostics.
Evaluate at a declared fixed interval, not in response to test scores.

The first candidate changes are epoch shuffling and class-balanced sampling,
tested as separate ablations with the same optimization budget and evaluation
protocol. A new objective such as conditional WGAN-GP is a later candidate only
if the baseline demonstrates a diagnosed stability/diversity problem; preregister
its configuration and budget before its comparison. Do not expand into an
unbounded architecture search merely to obtain a favorable graph.

Before full runs, freeze the available GPU-time budget, evaluation interval,
selection rule, and candidate configurations using the observed pilot throughput.
Record every attempted run, including failures and worse candidates. Stop on
nonfinite losses, incompatible resumes, invalid data, or exhaustion of the
declared budget. Resume later with a recorded continuation rather than quietly
changing the experiment.

For final comparisons, report all seeds and uncertainty, with paired comparisons
across common seeds. Use household-level resampling for intervals when household
groups exist, rather than treating correlated days as independent. Report the
number of independent seeds and households alongside 95% intervals and effect
sizes. A candidate is described as better only for the named metrics supported
by that comparison; profile shape, tails, and diversity trade-offs remain visible.

## 6. Figures and reusable research artifacts

Generate the figure families listed in the source audit. Retain orange real and
teal generated curves and the paper's seasonal/scenario arrangement when those
conditions exist. For the no-EV-label baseline, use season columns with a single
all-households row per day type; add PV facets only when verified PV metadata
exists. Do not relabel those panels with the paper's EV scenarios. Export PDF/SVG and at least 300-DPI PNG from the same saved
tables. Include units, split, counts, model/seed identifiers, and estimator version
in captions or accompanying metadata. Figures 1/2 are newly drawn workflow and
architecture diagrams, not copied page images.

Figures 7/8 use the paper's named classes when they exist. Otherwise choose and
freeze illustrative conditions using validation data before opening the test
results. Label unavailable original classes. Provide full per-class tables so
illustrative panels cannot conceal failures. Fix comparable axis/color ranges
across candidate models; disclose any saturation beyond a displayed range.

Each run directory contains configuration, data provenance, split assignments or
their privacy-appropriate audit, condition mapping, normalization parameters,
environment versions, Git SHA, seed, training log, checkpoints, generated samples,
metric tables, figure source tables, figures, and a machine-readable status.
Large datasets, checkpoints, and run outputs remain outside Git. Public reports
must not expose household identifiers or raw private records.

## 7. Software boundaries and Colab workflow

The package design separates ingestion/audit, splitting/preprocessing, model,
training/checkpointing, metrics, figure generation, and a thin command interface.
The notebook calls the same package used by automated tests; it does not contain
a second independent training implementation.

The Colab notebook will clone/check out `V2`, verify the chosen commit and GPU,
install only needed compatible project dependencies, download the declared data
asset, run the audit and split preparation, run a small integration check, train
or resume, evaluate, export figures, and persist artifacts to my selected Drive
storage. Mounting/authenticating Drive remains an explicit notebook interaction.
Do not run the existing CPU cloud installer over Colab's CUDA PyTorch.

Record and support the tested Colab Python/PyTorch versions rather than assuming
Colab matches this workspace's Python 3.12. Avoid GUI backends and secrets in
notebooks, logs, Git URLs, or saved configurations. Treat downloaded archive
paths as untrusted and reject traversal or unexpected extraction destinations.

## 8. Verification and delivery gates

Before model implementation, review this specification and then prepare a
file-level implementation plan. Implementation tests must cover:

1. Schema validation, units, duplicate keys, missing hours, finite values,
   legitimate negative net load, and missing condition metadata.
2. Household partition disjointness, deterministic split manifests, and
   train-only normalization/bin fitting with no test-data influence.
3. Model shapes, label validation, finite gradients, actual parameter updates,
   interrupted checkpoint recovery, and deterministic same-environment resume.
4. Metric analytical cases, permutation invariance, unequal sample counts,
   constant features, ties, empty classes, and invalid covariance inputs.
5. An end-to-end synthetic integration run explicitly labeled as a fixture,
   figure generation with inspectable tables, and notebook execution.
6. A measured-data pilot and a real Colab GPU run before claiming GPU workflow
   validation; full baseline/candidate runs before any performance claim.

Current deliverables are `V2`, source provenance, the completed release audit,
and this approved baseline design. The next data work is the long-format adapter,
declared load/calendar mapping, and daily-profile preparation. The GPU workflow
will run in Colab and preserve logs, checkpoints, metrics, and figures for review.
No research training has occurred and no same-or-better result is claimed.
