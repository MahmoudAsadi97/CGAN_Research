# Paper and reference-code audit

Reviewed on 2026-10-22. This is a source audit, not a model result.

## Sources and availability

- Paper: Li, Dong, and Qiu (2025), *Conditional generative adversarial network
  (cGAN) for generating building load profiles with photovoltaics and electric
  vehicles*, Energy and Buildings 335, 115584.
  DOI: <https://doi.org/10.1016/j.enbuild.2025.115584>.
- Reference: <https://github.com/malvin0130/cGAN_EB>, inspected at
  `885f78052a91fbc755e6b17a354c68849b5556ca`, MIT licensed.
- Working repository: <https://github.com/MahmoudAsadi97/CGAN_Research>, branch
  `V2`. I keep `main` unchanged until I decide to merge.
- [Machine-readable provenance and numerical probes](source-manifest.json).
- [Upstream MIT license](upstream-MIT-LICENSE.txt). Retain this attribution when
  adapting its implementation.

I identify the source PDF by its DOI and SHA-256 in the provenance manifest.
The PDF itself is not redistributed in this public repository.

At inspection, my `Dataset` release contained only GitHub's automatic
source archives. Its tag points to the initial setup commit, which contains
`docs/cloud-environment.md` and `scripts/setup-cloud.sh`. No attached measured
dataset or dataset link in a release description was observed. Recheck the
release when an actual asset is supplied; absence here is a dated observation.

The upstream repository includes 64 synthetic CSVs, each with 2,000 rows and 24
hourly columns. They are **generated examples**, not the original measured data
or my measured dataset. `train.py` expects `new_train_all.csv`, which is not
included. The paper's data-availability statement says its data is confidential.
Exact numerical reproduction of the paper is therefore not currently possible.

## Method described by the paper

The research task is conditional generation of daily electricity-load profiles, not image
generation or forecasting a specific measured day. The original observations
cover 110 households, 2013–2019, with 24 hourly readings per daily profile. The
paper reports 147,245 weekday profiles and 58,764 weekend profiles.

Its 64 labels combine eight adoption/charger scenarios, four seasons, and two
day types. Holidays are grouped with weekends. The eight scenarios are no PV/EV,
PV only, EV level 1, EV level 2, EV with both chargers, and PV plus each of those
three EV charger categories. These labels are justified only when the input data
actually establishes the corresponding status on that date.

The published settings are a 24-dimensional standard-normal noise vector,
one-hot condition encoding, hidden width 64, LeakyReLU activations, discriminator
dropout 0.4, Adam, generator learning rate 0.0001, discriminator learning rate
0.0002, batch size 2,000, 10,000 epochs, and 2,000 generated profiles per class.

## Implementation and evaluation discrepancies

All source links below refer to the inspected commit, not a moving branch.

| Area | Observed implementation | Consequence for V2 |
| --- | --- | --- |
| Data split | [`train.py`, lines 8–46](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/train.py#L8) trains on the loaded data and writes those same profiles as the real comparison set; no held-out split is implemented. | Report a genuine held-out evaluation and identify any training-set comparison explicitly. |
| Normalization | Paper section 2.2 describes division by a fixed dataset maximum. `train.py`, lines 16–22, applies global min–max normalization. | Implement named normalization modes; fit parameters only on the training partition for the primary evaluation. |
| Negative values | `train.py`, lines 13–14, discards profiles outside [0, 100]. | Do not silently discard legitimate negative net-export readings or assume the same units and bounds for the new dataset. |
| Network description | [`cGAN.py`, lines 10–43](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/cGAN.py#L10) implements G: `(24+K)→64→128→64→24`; D: `(24+K)→64→128→256→1`. Table 2 also lists an 88-unit layer, apparently the concatenated input when K=64. | Use the released code's exact dimensions for the reference architecture and disclose the table interpretation. |
| Training order and history | `cGAN.py`, lines 74–115, does not shuffle between epochs and accumulates loss history across all epochs. | Make shuffle policy explicit; log per-epoch losses and optimizer steps. An improved shuffled variant must not be described as an exact training-loop reproduction. |
| Near-base / near-peak | Paper section 2.4 specifies the 2.5th / 97.5th percentiles; [`utils.py`, lines 157–174](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/utils.py#L157) selects sorted indices 1 and 22. | Keep a named compatibility calculation separate from interpolated quantiles. |
| Rise / fall times | `utils.py`, lines 162–179, uses absolute index differences between extrema of distance to a threshold. The paper describes chronological transitions. | Do not silently treat those definitions as equivalent; document and test both a compatibility definition and an explicit temporal definition. |
| KL divergence | [`utils.py`, lines 274–290](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/utils.py#L274) applies `rel_entr` to scaled sample values, not normalized probability masses. Pandas indices are retained, and nonfinite terms are discarded. | Use shared-bin probability distributions for primary KL. Label the upstream calculation as a legacy diagnostic, never as a valid primary divergence. |
| “FID” | [`utils.py`, lines 102–117](https://github.com/malvin0130/cGAN_EB/blob/885f78052a91fbc755e6b17a354c68849b5556ca/utils.py#L102) and the final plotting block compute a separate Gaussian Fréchet distance for each handcrafted scalar feature. No Inception network is used. | Call these feature-space Fréchet distances. Report scalar and joint-feature results separately and state units/scaling. |
| Fixed class assumptions | Output routines hard-code 64 names and class numbers. | Derive supported classes from verified metadata; mark unavailable combinations instead of inventing data or filling metrics with zeros. |
| Plotting | `stats_plot` selects `TkAgg`. | Use a headless backend for cloud and Colab export. |

### Numerical checks performed during review

For the daily vector `0,1,...,23`, linear 2.5th/97.5th quantiles are 0.575 and
22.425, whereas the released order-statistic code returns 1 and 22.

The released KL-like expression was evaluated on real samples `[1,2,3,4]` and
generated samples `[1,1,4,4]`. It returned approximately **0.032703**. Permuting
only the generated row order to `[4,4,1,1]` returned **0.379277**. The empirical
distributions did not change. A distribution-level divergence must be invariant
to that permutation. This probe establishes an implementation problem; it does
not establish how much any published figure would change without the original
real dataset and complete experimental provenance.

## Figure inventory

| Paper figure | Page | Required regenerated output |
| --- | --- | --- |
| 1 | 3 | Original diagram of the implemented data→training→evaluation workflow, with attribution to the paper's method. |
| 2 | 5 | Original architecture diagram with actual tensor dimensions and condition encoding. |
| 3, 4 | 7, 8 | Weekday/weekend grids of hourly real and generated mean profiles: eight scenario rows × four season columns when supported. |
| 5, 6 | 9 | Weekday/weekend heatmaps of signed generated-minus-real differences in five feature means and standard deviations. |
| 7 | 10 | Near-peak distribution comparisons, four declared conditions. |
| 8 | 10 | Near-base distribution comparisons, four declared conditions. |
| 9, 10 | 11 | Weekday/weekend per-feature KL heatmaps, with the estimator explicitly identified. |
| 11, 12 | 12 | Weekday/weekend per-feature Gaussian Fréchet-distance heatmaps. |

Paper pages 7, 9, 10, 11, and 12 were rendered for figure inspection. Figures
3 and 5/6 were visually inspected. The remaining layouts were checked against
their captions and plotting code. My dataset has no EV labels. V2 uses verified calendar conditions for the
primary baseline and does not reproduce the paper's EV-specific class claims.
PV status is still unverified. Any inferred EV labels require a separate
validation protocol and cannot be described as measured EV status.

New plots must be generated from saved result
tables, not traced from paper curves. Use consistent axes and disclose clipped
color ranges. If the new dataset lacks original conditions, adapt and label the
layout instead of implying a 64-class reproduction.

## Meaning of “same or better”

The paper reports KL around 0.3 or below, most feature Fréchet distances below 5,
and small feature mean/std differences. These are contextual reference values,
not universal acceptance thresholds: metric definitions, data distributions,
units, sample counts, and evaluation splits affect them.

The defensible primary comparison is the reference cGAN architecture versus
candidate improvements on the **same new dataset, frozen splits, estimator
definitions, generation counts, and declared seeds**. Present raw-paper-style
diagnostics separately from corrected metrics. A lower score on a different
dataset does not establish superiority over the paper's original experiment.

Current status: branch created and pushed; source review and numerical probes
completed; actual dataset unavailable; no research model trained or evaluated.
