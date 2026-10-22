# Dataset availability and quality audit

Checked on 22 October 2026. I now have all eight electricity files, the source
data dictionary, and the temperature workbook in the supplied release. This
audit establishes file integrity and observed coverage; no model has been trained.

## Source and integrity

I downloaded [`raw.zip`](https://github.com/MahmoudAsadi97/CGAN_Research/releases/download/Dataset/raw.zip)
from the `Dataset` release: **395,824,014 bytes**. Its SHA-256 matches the digest
published in the release asset listing:

```text
854b18f4adfd269675baabe7388578f4a1b7bb440529c09854fa0952574ae449
```

All eight electricity CSVs were streamed completely, including ZIP CRC checks.
The weather workbook and legend were read in full. The release weather file is
byte-for-byte identical to the separately uploaded workbook. The archive also
contains a preparation log; I verified coverage from the CSVs rather than relying
on that log. [Aggregate results and inventory](dataset-audit.json) record the
checks without publishing individual meter records.

## Electricity coverage

The data contains **84,326,400 quarter-hour rows from 2,400 distinct meters** for
2024. Each of the eight cohorts has **300 meters and 10,540,800 rows**. All 28
pairs of cohort files have disjoint meter IDs. Meter IDs are grouping keys; I
do not assume an independently established one-to-one household mapping.

| File suffix | Heat-pump flag | EV-detection flag | PV flag | Import maximum, kWh/interval | Export maximum, kWh/interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| `enkel_ZP` | 0 | 0 | 1 | 3.267 | 1.968 |
| `geen_ZP` | 0 | 0 | 0 | 5.144 | 0.001 |
| `WP_geen_ZP` | 1 | 0 | 0 | 6.220 | 0.178 |
| `WP_met_ZP` | 1 | 0 | 1 | 5.632 | 2.466 |
| `EV_geen_ZP` | 0 | 1 | 0 | 4.874 | 0 |
| `EV_met_ZP` | 0 | 1 | 1 | 5.218 | 2.797 |
| `WP_EV_geen_ZP` | 1 | 1 | 0 | 7.095 | 0.001 |
| `WP_EV_met_ZP` | 1 | 1 | 1 | 6.246 | 3.007 |

Every file has the same nine fields: meter ID, date, interval start, imported
energy, exported energy, heat-pump indicator, EV-detection indicator, PV indicator,
and contract category. All contracts are `Residentieel`. Indicators are constant
within each meter and agree with their file cohorts. The full scan found no null
fields, unparseable/nonfinite/negative import or export readings, off-grid or
out-of-year timestamps, or mismatches between the date and written timestamp.

The legend defines both energy fields in **kWh during the following quarter-hour**.
They are not instantaneous kW. Hourly energy requires summing four intervals;
quarter-hour average power in kW is four times interval energy in kWh. Imported
energy minus exported energy gives net grid exchange, which can be negative.
This does not establish behind-the-meter gross consumption or total PV production.

## Time and daylight saving

Each meter has 35,136 rows from January 1 00:00 to December 31 23:45. The complete
occurrence pattern matches the documented Belgian CET/CEST year:

- March 31 has 92 rows: local 02:00, 02:15, 02:30, and 02:45 do not occur.
- October 27 has 100 rows: those four local labels occur twice.
- Each other date has 96 rows.

The CSV appends a literal `Z`, which conflicts with its legend and observed clock
changes. I therefore preserve the written wall-clock labels and do not interpret
that suffix as established UTC. The two autumn occurrences are not accidental
duplicates to drop. This audit verifies their counts, not the assignment of each
reading to a UTC instant. The initial 24-hour profile protocol excludes both
transition dates and records those exclusions; this would leave 873,600 meter-days
before any other quality filtering or train/validation/test split.

## Temperature and remaining metadata

`Brussels_Temperature_15min_2024.xlsx` contains **35,136 rows**, covering the entire
2024 naive quarter-hour grid. The columns are `datetime` and `T_out`. There are
no missing values, duplicate timestamps, or nonfinite temperatures. Values range
from -6.769869 to 33.950468, in an **undocumented unit**.

The workbook has 96 rows even on both DST dates. Its timestamps have no timezone;
the file does not establish UTC, Belgian local time, or fixed standard time. A
direct join is therefore not validated, and simply removing the two DST days
does not resolve a possible year-round clock offset. Before using weather I need
its timezone, units, source/station, observation versus interpolation method,
and timestamp/interval convention. I also need to justify using this Brussels
series as a weather proxy for the sampled meters. The electricity-only baseline
can proceed without weather once its preprocessing and calendar are implemented.

## Label interpretation and source caveats

The release already includes `Elektrisch_Voertuig_Indicator`. The legend includes
battery and plug-in hybrid vehicles: `1` means **detected**, and `0` means
**not detected**, which explicitly does not prove absence. I do not need to create
new labels to describe those two cohorts. This field does not provide independently
verified ownership, adoption dates, charger levels, or charging-event ground truth.
The paper's eight charger/adoption scenarios are different from these eight cohorts.

The heat-pump indicator is described as reported main heating, but its boolean
descriptions contain apparent EV-related copy-paste text. I preserve the supplied
flag and record that documentation inconsistency. The PV indicator denotes a
reported installation, but **352 positive export rows occur where PV=0**: one in
`geen_ZP`, 339 in `WP_geen_ZP`, and 12 in `WP_EV_geen_ZP`. I do not silently change
the flags or zero those readings; their cause remains unresolved.

The legend describes selected residential meter cohorts with a full year of
readings; known storage systems are excluded, but unknown storage may remain.
Equal cohort sizes do not estimate technology prevalence in the wider population.
Full file coverage does not independently verify measurement accuracy or justify
publication claims about model performance.

## Repeating the checks

The probes were checked with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, and
openpyxl 3.1.5. Supply local paths; a GitHub release page is not a filesystem
directory.

```bash
python scripts/audit_release_electricity.py \
  --archive /path/to/raw.zip --output /path/outside/repository/electricity-audit

python scripts/audit_temperature.py \
  --workbook /path/to/Brussels_Temperature_15min_2024.xlsx \
  --output /path/outside/repository/temperature-audit
```

The electricity probe reads all eight CSVs by default; `--part1` or `--part2`
selects one half. It records per-meter occurrence arrays for coverage and
cross-file identity checks. Its missing/repeated-slot counts use a naive grid
and include the expected DST changes described above. Keep those local artifacts,
raw data, and internal meter IDs outside Git. The committed results contain only
aggregate counts and provenance. The independent two-part full scan was reconciled
across all eight files; the portable temperature probe was replayed against the
uploaded workbook with identical numerical and timestamp findings.

A separate eight-file synthetic replay checked the portable electricity command,
cross-file overlap detection, DST counts, and detection of invalid numbers,
infinity, negative readings, and a malformed date. These fixture checks validate
the probe; they are not measured-data or model-performance results.
