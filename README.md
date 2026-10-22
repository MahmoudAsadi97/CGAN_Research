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

- The development branch is `V2`. I will decide when to merge it; `main` remains
  unchanged.
- The research design is approved, with EV labels excluded from the baseline.
- The implementation plan is available for review. The training pipeline has
  not been implemented yet.
- The `Dataset` release now includes `raw.zip`. I downloaded it and verified
  its SHA-256 against the published release checksum. The archive contains
  eight electricity CSVs, a data dictionary, and the Brussels temperature
  workbook. The [dataset audit](docs/research/dataset-audit.md) records coverage,
  label definitions, and the remaining weather-alignment questions.
- The CPU development environment has passed synthetic dependency checks. No
  model has been trained or evaluated on my measured data, and no improvement
  over the paper is claimed.

## Documentation

- [Research design](docs/research/v2-design.md)
- [Implementation plan](docs/research/implementation-plan.md)
- [Paper and source-code audit](docs/research/source-audit.md)
- [Source provenance](docs/research/source-manifest.json)
- [Dataset availability and quality audit](docs/research/dataset-audit.md)
- [EV-labeling considerations](docs/research/ev-labeling.md)
- [CPU development environment](docs/cloud-environment.md)

The existing `scripts/setup-cloud.sh` is for the CPU cloud workspace. I do not
run it over Colab's GPU-enabled PyTorch installation. The planned Colab notebook
will run the same research package and preserve logs, checkpoints, evaluation
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
