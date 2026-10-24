# Electricity-only V2 experiment register

## Scope and source identity

I worked in the existing `V2` checkout starting at
`b5d91c99886a70792b9100566e3cf45546109f2e`. I have not changed `main` or
rewritten published history. My implemented baseline uses season and calendar
day type, with no temperature or technology conditions. I preserve provider
EV detection as supplied metadata: `1` means detected BEV/PHEV, and `0` means
not detected, without establishing absence, ownership, or charging events.

I used the existing `gpu_env`, with Python 3.10.12, PyTorch 2.5.1 / CUDA build
12.1, NumPy 1.26.4, and pandas 2.3.3. Runtime records also capture dependency
versions, Git SHA/dirty state, and a package-source SHA-256. I have not established
compatibility with other Python, GPU, or Colab environments through this run.

My local `/mnt/c/Users/Nima/Downloads/raw.zip` contains the eight electricity
cohorts at archive root and omits the weather workbook. I record its identity as
`local_variant_unverified_release`:

| Source | Bytes | SHA-256 |
| --- | ---: | --- |
| Local archive used | 395,085,174 | `934cc871e9fe18dbe30530472816d711ece878ec7cd8725d9d8417bbdb389bd8` |
| Previously audited published release | 395,824,014 | `854b18f4adfd269675baabe7388578f4a1b7bb440529c09854fa0952574ae449` |

I do not claim these archives or their electricity files are byte-equivalent.
My current checks establish the local archive's schema, coverage, accepted-day
integrity, and source-to-hour aggregation. I retain its actual checksum together
with the published reference checksum and mismatch status. I finalized this
provenance before freezing the new prepared artifact.

## Measured intake on 23 October 2026

I streamed all eight electricity CSVs to EOF, including ZIP CRC validation.
The intake and preparation took approximately 590 seconds in this session.

| Quantity | Observed result |
| --- | ---: |
| Quarter-hour records read | 84,326,400 |
| Distinct meter groups | 2,400 |
| Accepted 24-hour daily profiles | 873,600 |
| Excluded meter-days on March 31 | 2,400 |
| Excluded meter-days on October 27 | 2,400 |
| Other exclusions | 0 |
| Training profiles / meters | 611,520 / 1,680 |
| Validation profiles / meters | 131,040 / 360 |
| Test profiles / meters | 131,040 / 360 |

I sum four quarter-hour imported-minus-exported kWh readings per hour. My output
is net grid exchange, not gross consumption or total PV production. I retain
negative net values and all supplied metadata without changing flags from load
shapes. I interpret the literal timestamp `Z` according to the documented
Belgian local wall-clock convention and quarantine both DST transition days.

I independently compared 824 complete meter-days from the first 10,000 rows of
each cohort against the prepared hourly arrays. The largest absolute hourly
difference was `3.552713678800501e-15` kWh. This is a source-aggregation check,
not an independent validation of the underlying measurements.

I froze group assignments using seed 42 and 70/15/15 fractions. Meter identifiers
are grouping proxies; I have not independently verified one meter per household.
All eight calendar classes occur in training and both held-out partitions.
My training-only min–max transform has minimum `-10.418000000000001` kWh and
scale `33.268` kWh. I preserve 17 out-of-range hourly values in test and zero
in validation; those are intake diagnostics, not model scores or tuning inputs.
All five training features are nonconstant. My calendar uses the declared
season month groups, Saturday/Sunday weekends, and no supplied holidays.

I verified the serialized NPZ, array-content, and manifest hashes after saving:

- Data fingerprint: `9e17aae84c4efa889c20a0479ad7bb72008bc8c335d83f8ba5aa34585acd2807`.
- Split fingerprint: `98f0bbfd84b8fc4b51ce94f189fbca0beae24c40895613e6cc0f3ff50f1e4ecd`.

I keep private data, meter identifiers, and generated artifacts out of Git:
`outputs/measured-2024/` contains the prepared arrays, private manifest,
aggregate audit, independent source check, and training-only metric reference.
Public evaluation JSON and figure source tables use an explicit provenance
allowlist; grouped records for uncertainty remain in `private_records.npz`.

## Implementation and independent review

I implemented the reference dense architectures and Adam settings, detached
fake profiles for the discriminator update, froze discriminator parameters for
the generator update, and used stable logits/BCE losses. I retain fixed order as
the baseline, with shuffle and balanced sampling named separately. I save
step histories, sample-weighted epoch losses, complete optimizer/RNG/sampling
state, configuration, normalization, and data/split identity. Checkpoint writes
are atomic; selected checkpoints have immutable step-addressed files, and
resume repairs the convenience `best.pt` copy when publication was interrupted.

I select checkpoints only using fixed-noise validation macro feature Wasserstein
distance divided by training feature standard deviations. I exclude exact
training-constant features from selection and retain their errors in reports.
Generated samples carry their physical units, data/split identity, condition
mapping, training seed, and separate sampling seed. Evaluation rejects mismatches.
I have not used the frozen test set for model selection or model evaluation.

I implemented corrected quantile/cyclic features, smoothed shared-bin KL/JS,
Wasserstein, scalar/joint Gaussian Fréchet, hourly mean-profile RMSE, energy/ramp,
autocorrelation, diversity, and nearest-training-profile diagnostics. I keep
historical feature definitions separate and retain upstream MIT attribution.
My joint Fréchet computation uses PSD square roots and an equivalent singular-
value trace for stability with singular covariances. I report 15/30/60-bin
sensitivity and 100 independent real-resampling references without selecting the
best-looking setting. I save all generated samples without quality filtering.

I implemented paired-seed summaries and whole-meter/household-group bootstrap
intervals for the scaled macro feature Wasserstein effect. The group intervals
are conditional on the evaluated seeds; they are separate from seed intervals,
not a joint two-level uncertainty estimate. Other metric summaries do not gain
household intervals merely because this selection-score interval is available.
Missing private records, unsupported conditions, and unavailable cells are
reported explicitly. Comparison checks do not depend on private-cache presence.

I obtained independent reviews of training, data/metrics, and the integrated
pipeline. I addressed the important findings with failing regression tests
followed by passing tests:

- I configured deterministic CUDA's cuBLAS prerequisite before device use.
- I made selected-checkpoint publication recoverable across interruptions.
- I detected exact constant features by range rather than rounded standard deviation.
- I removed private meter metadata from public evaluation provenance.
- I included unsupported conditions in unavailable cells and exclusion counts.
- I integrated paired group resampling into saved-run comparisons.
- I rejected generated samples from a different split or condition mapping.
- I checked shared sampling seeds, references, and sample counts before comparisons.

I also corrected the figure-manifest assertion, retained published-source
provenance keys, rejected nonfinite optimizer decay, separated training and
sampling seeds, and recorded source/dependency fingerprints. My synthetic CPU
checks cover exact mid-epoch resume in all three sampling modes across validation,
physical-unit generation, analytical metric cases, a two-epoch CLI path, and the
executable notebook smoke. Figure tests check source data, curve coordinates,
missing cells, tails/saturation, and PDF/SVG/300-DPI PNG exports.

## Final CPU verification

I ran the complete suite on the final implementation:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MPLCONFIGDIR=/tmp/cgan-mpl \
OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  --basetemp=outputs/verification-tests
```

My result was **62 passed in 103.82 seconds**, with no failures. I disabled
external pytest plugin autoload because this environment's unrelated ROS
`launch_testing` plugin is incompatible with the installed pytest hooks. I did
not replace the environment or remove its packages. The exact test output is
in `outputs/verification-suite.log`; synthetic integration checkpoints and
figure exports are under `outputs/verification-tests/`.

I visually inspected the synthetic seasonal-profile export and checked its
orange/teal series, units, counts, split, training/sampling seeds, and synthetic
caption. These plots deliberately retain the poor short-run samples; they
provide software verification, not model-quality evidence. My Git whitespace
check passed, and my unignored-file inventory contains only source, tests,
configuration, documentation, and the unexecuted notebook file.

I retain a minor checkpoint limitation from review: loading validates safe
serialization, version, and expected identities, while network shape checks
occur when weights load. I do not yet provide a complete standalone schema
validator for every malformed training-state field. Exact GPU resume remains
unverified; the interrupted/resume equivalence tests ran on CPU.

## Bounded GPU attempt

I attempted the actual packaged training command on the prepared measured data:

```bash
PYTHONPATH=src python -m cgan_research train \
  --prepared outputs/measured-2024 \
  --config configs/calendar_baseline.yaml \
  --output outputs/pilot-cuda-20261023
```

I fixed the pilot budget before that attempt: seed 17, fixed order, batch size
2,000, at most 400 optimizer steps, at most 600 cumulative seconds, validation
every 100 steps using 2,000 generated profiles per condition, and checkpoints
every 25 steps. Time checks occur between batches; validation/checkpoint work can
overrun the wall-clock deadline. The 10,000-epoch reference ceiling does not
implicitly launch a full experiment.

The command exited **2**, with `CUDA is unavailable; no CPU fallback is allowed`.
My saved `outputs/pilot-cuda-20261023/status.json` reports **failed, 0 steps**.
No measured-data model checkpoint was created. I do not report GPU throughput,
GPU memory usage, trained-model quality, or a completed GPU pilot. My earlier
`gpu-check.json` describes a different accessible session; currently `/dev/dxg`
is absent and NVIDIA reports GPU access blocked by the operating system.

## Delivery limits and continuation

I attempted Git staging, but `.git` is mounted read-only and `git add` could not
create `index.lock`. GitHub access also failed DNS resolution. I cannot commit
or push from this managed session, and I have not bypassed those restrictions.
My tested changes remain in the V2 working tree; no new commit or push is claimed.

I need a session with GPU device access, writable repository metadata, and GitHub
connectivity to finish the requested delivery. I will retain ordinary descriptive
commit messages without co-author trailers. After inspecting the final diff and
Git exclusions, my continuation is to commit/push V2 and launch the same bounded
pilot in a fresh output directory. A failed preflight has no checkpoint to resume.
I will use measured GPU throughput to preregister a later full-run budget.

I have not executed the five-seed baseline/candidate comparison, selected a
candidate using test results, or verified a hosted Colab GPU/Drive restart.
I have not established verified household identity, EV detector accuracy,
measurement accuracy, population representativeness, or privacy guarantees.
The implementation and intake evidence do not establish improvement,
paper-equivalent performance, or publication readiness.
