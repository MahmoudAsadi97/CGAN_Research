# EV labels: evidence, uncertainty, and research claims

The inspected release includes `Elektrisch_Voertuig_Indicator`, a provider-supplied
EV-detection flag. According to its data dictionary, it includes battery and
plug-in hybrid vehicles; `1` means detected and `0` means not detected, which
does not establish absence. These labels can define detected/not-detected
cohorts without creating new labels. They do not establish verified ownership,
charger level, adoption date, charging events, or isolated EV consumption.

I keep the approved calendar-conditioned baseline independent of EV inference.
A cohort-conditioned extension should preserve the provider label and its
limitations. The separate considerations below apply if I later develop or
validate my own detector. See the [dataset audit](dataset-audit.md).

## Define what a label means

| Target | Suitable independent reference | What it does not establish |
| --- | --- | --- |
| Household owns an EV on a given date | Ownership/adoption records with dates | Whether the vehicle charged at home that day |
| EV charging occurred during an interval | Synchronized charger logs, EV submeter readings, or vehicle charging records | Ownership status on every other day |
| Load resembles a charging pattern | A declared heuristic or detector applied to aggregate readings | Verified EV ownership, actual charging, or isolated EV consumption |
| Synthetic charging was added | A documented simulation and its injection records | Detection performance on real EV events |

No detected charging event is not a verified no-EV label. A household can own
an EV without charging it at home on a particular day. A high-load interval can
also come from heating, cooling, cooking, or other appliances. Hourly aggregate
readings can hide event duration and power transitions.

## Optional inferred-pattern experiment

Before implementing a detector, I need to inspect the actual sampling frequency,
load units, household/date identifiers, net-versus-gross load meaning, missingness,
and any independent records. I will then define one target, a labeling method,
and a validation protocol. A visual interpretation of the same load curve is
not independent ground truth.

If labels are inferred, I will preserve the target, label source, detector score,
threshold, and an uncertain/abstain state. An uncalibrated score will not be
reported as the probability of EV charging. Households or intervals without
enough evidence remain unknown instead of being forced into a negative class.

I will split households and relevant time periods before fitting preprocessing,
rules, or a learned detector. Detector selection and thresholds use training
and validation data, never the final test results. Independently labeled test
data can support precision, recall, false-positive rate, calibration where
appropriate, and uncertainty across households. Threshold sensitivity and the
fraction of abstained cases must be reported alongside performance.

Running the same rule on generated profiles only measures agreement with that
rule. It cannot independently establish that generated profiles represent EV
charging. Labels derived from the generated target's load shape can define a
descriptive load-pattern category, but do not establish physical EV semantics
or an input available for prospective forecasting.

## Permitted interpretation

Without independent validation, I can describe a model as generating profiles
conditioned on **inferred load-pattern categories**. I cannot claim verified
EV-specific generation, successful real EV detection, EV energy disaggregation,
or the causal effect of adding an EV.

Any optional model must also be compared with the no-EV-label baseline using the
same held-out data and general load-quality metrics. The EV-related study will
require a separate protocol decision after the available evidence is known.
