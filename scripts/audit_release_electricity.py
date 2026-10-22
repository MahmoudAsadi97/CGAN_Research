"""Stream the eight 2024 electricity CSVs and write local audit artifacts.

Usage: python scripts/audit_release_electricity.py --archive /path/to/raw.zip \
    --output /path/outside/repository/audit-results [--part1 | --part2]

Requires pandas and numpy. Reads every selected ZIP member to EOF, validating
its CRC. Aggregates and compressed coverage arrays include internal meter IDs;
keep all output outside the repository. Console progress contains only counts.

The source legend documents Belgian local CET/CEST clock labels even though
the CSV appends Z. The probe checks written clock fields without inferring UTC
instants or silently deduplicating the autumn clock change. Missing/duplicate
coverage metrics use a naive 366-day grid and therefore include DST effects.
"""

import argparse
import collections
import csv
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd

EXPECTED = ['EAN_ID', 'Datum', 'Datum_Startuur', 'Volume_Afname_KWh', 'Volume_Injectie_KWh', 'Warmtepomp_Indicator', 'Elektrisch_Voertuig_Indicator', 'PV_Installatie_Indicator', 'Contract_Categorie']
LABELS = EXPECTED[5:]
READINGS = EXPECTED[3:5]
NS = 900_000_000_000
START = pd.Timestamp('2024-01-01T00:00:00').value
SLOTS = 366 * 96

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def summarize_counts(values):
    u, n = np.unique(values, return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u, n)}

def iso(ns):
    # Reproduce the literal source suffix, not an assertion that the label is UTC.
    return pd.Timestamp(int(ns), unit='ns').isoformat(timespec='milliseconds') + 'Z' if ns is not None else None

def audit(z, info, output):
    started = time.monotonic()
    stem = Path(info.filename).stem
    out = {'member': info.filename, 'uncompressed_bytes': info.file_size, 'zip_crc32': f'{info.CRC:08x}', 'expected_schema': EXPECTED, 'rows': 0, 'null_counts': collections.Counter(), 'numeric': {}, 'labels': {}, 'timestamps': {}, 'datum': {}}
    with z.open(info) as raw:
        out['actual_schema'] = next(csv.reader([raw.readline().decode('utf-8-sig').rstrip('\r\n')]))
    if out['actual_schema'] != EXPECTED:
        raise RuntimeError('Schema mismatch: ' + info.filename)
    ids, idmap = [], {}
    occurrence = np.zeros((0, SLOTS), dtype=np.uint32)
    meter_rows = np.zeros(0, dtype=np.int64)
    meter_ts_min = np.zeros(0, dtype=np.int64)
    meter_ts_max = np.zeros(0, dtype=np.int64)
    meter_labels = []
    label_counts = {c: collections.Counter() for c in LABELS}
    dtypes = {c: set() for c in EXPECTED}
    numeric = {c: {'null_or_nan': 0, 'parse_invalid_nonnull': 0, 'positive_infinity': 0, 'negative_infinity': 0, 'negative_finite': 0, 'zero': 0, 'finite_count': 0, 'min_finite': None, 'max_finite': None} for c in READINGS}
    ts_invalid = ts_noncanonical = ts_offgrid = ts_outside = ts_null = datum_invalid = datum_mismatch = 0
    global_min = global_max = None
    global_date_min = global_date_max = None
    unusual_keys = collections.Counter()
    with z.open(info) as raw:
        reader = pd.read_csv(raw, dtype='string', chunksize=250_000, keep_default_na=False, na_values=[''])
        for chunkno, chunk in enumerate(reader, 1):
            out['rows'] += len(chunk)
            for c in EXPECTED:
                out['null_counts'][c] += int(chunk[c].isna().sum())
                dtypes[c].add(str(chunk[c].dtype))
            # Internal IDs are retained only in local NPZ/JSON files, never printed.
            key_ids = chunk['EAN_ID'].fillna('<NULL_EAN_ID>')
            local_codes, local_ids = pd.factorize(key_ids, sort=False)
            for mid in local_ids:
                if mid not in idmap:
                    idmap[mid] = len(ids)
                    ids.append(str(mid))
                    meter_labels.append({c: set() for c in LABELS})
            extra = len(ids) - len(occurrence)
            if extra:
                occurrence = np.concatenate((occurrence, np.zeros((extra, SLOTS), dtype=np.uint32)))
                meter_rows = np.pad(meter_rows, (0, extra))
                meter_ts_min = np.concatenate((meter_ts_min, np.full(extra, np.iinfo(np.int64).max, dtype=np.int64)))
                meter_ts_max = np.concatenate((meter_ts_max, np.full(extra, np.iinfo(np.int64).min, dtype=np.int64)))
            codes = np.asarray([idmap[mid] for mid in local_ids], dtype=np.int32)[local_codes]
            meter_rows += np.bincount(codes, minlength=len(ids))
            for c in LABELS:
                v = chunk[c].fillna('<NULL>')
                label_counts[c].update({str(k): int(n) for k, n in v.value_counts(dropna=False).items()})
                for mid, val in pd.DataFrame({'code': codes, 'val': v.to_numpy()}).drop_duplicates().itertuples(index=False, name=None):
                    meter_labels[int(mid)][c].add(str(val))
            for c in READINGS:
                v = pd.to_numeric(chunk[c], errors='coerce').to_numpy(dtype=np.float64, na_value=np.nan)
                finite = np.isfinite(v)
                stats = numeric[c]
                stats['null_or_nan'] += int(np.isnan(v).sum())
                stats['parse_invalid_nonnull'] += int((np.isnan(v) & chunk[c].notna().to_numpy()).sum())
                stats['positive_infinity'] += int(np.isposinf(v).sum())
                stats['negative_infinity'] += int(np.isneginf(v).sum())
                stats['negative_finite'] += int((finite & (v < 0)).sum())
                stats['zero'] += int((finite & (v == 0)).sum())
                stats['finite_count'] += int(finite.sum())
                if finite.any():
                    lo, hi = float(v[finite].min()), float(v[finite].max())
                    stats['min_finite'] = lo if stats['min_finite'] is None else min(lo, stats['min_finite'])
                    stats['max_finite'] = hi if stats['max_finite'] is None else max(hi, stats['max_finite'])
            rawts = chunk['Datum_Startuur']
            canonical = rawts.str.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.000Z', na=False).to_numpy(dtype=bool)
            # Z is a literal in this format. Source legend documents CET/CEST;
            # retain naive wall-clock labels rather than infer UTC instants.
            parsed = pd.to_datetime(rawts, format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
            # Datetime arrays may use microseconds in newer pandas releases.
            # Normalize the resolution before using nanosecond grid arithmetic.
            ns = parsed.dt.as_unit('ns').array.asi8
            valid = ns != np.iinfo(np.int64).min
            ts_invalid += int((~valid).sum())
            ts_noncanonical += int((~canonical).sum())
            ts_null += int(rawts.isna().sum())
            if valid.any():
                lo, hi = int(ns[valid].min()), int(ns[valid].max())
                global_min = lo if global_min is None else min(global_min, lo)
                global_max = hi if global_max is None else max(global_max, hi)
                np.minimum.at(meter_ts_min, codes[valid], ns[valid])
                np.maximum.at(meter_ts_max, codes[valid], ns[valid])
            # Require calendar parsing and exact alignment; neither suffix nor row order is trusted.
            slots = np.zeros(len(chunk), dtype=np.int64)
            slots[valid] = (ns[valid] - START) // NS
            aligned = np.zeros(len(chunk), dtype=bool)
            aligned[valid] = (ns[valid] - START) % NS == 0
            inside = (slots >= 0) & (slots < SLOTS)
            grid = valid & aligned & inside
            ts_offgrid += int((valid & ~aligned).sum())
            ts_outside += int((valid & ~inside).sum())
            np.add.at(occurrence, (codes[grid], slots[grid]), 1)
            for mid, t in zip(codes[~grid], rawts[~grid].fillna('<NULL>')):
                unusual_keys[(int(mid), str(t))] += 1
            rawdate = chunk['Datum']
            dates = pd.to_datetime(rawdate, format='%Y-%m-%d', errors='coerce')
            valid_date = dates.notna().to_numpy()
            strict_date = rawdate.str.fullmatch(r'\d{4}-\d{2}-\d{2}', na=False).to_numpy(dtype=bool)
            datum_invalid += int((~valid_date | ~strict_date).sum())
            datum_mismatch += int((rawdate.fillna('<NULL>') != rawts.str.slice(0, 10).fillna('<NULL_TS>')).sum())
            if valid_date.any():
                lo, hi = dates.min().isoformat(), dates.max().isoformat()
                global_date_min = lo if global_date_min is None else min(global_date_min, lo)
                global_date_max = hi if global_date_max is None else max(global_date_max, hi)
            if chunkno % 10 == 0:
                print(json.dumps({'progress_file': stem, 'rows': out['rows'], 'meters_so_far': len(ids), 'elapsed_s': round(time.monotonic()-started, 1)}), flush=True)
        # Exhaustion validates ZIP CRC; this extra read also asserts full stream EOF.
        assert raw.read(1) == b''
    missing = occurrence == 0
    duplicate = occurrence > 1
    missing_meter = missing.sum(axis=1)
    duplicate_meter = duplicate.sum(axis=1)
    extra_meter = np.where(duplicate, occurrence - 1, 0).sum(axis=1)
    present_meter = (~missing).sum(axis=1)
    anomalies = []
    for slot in np.flatnonzero(missing.any(axis=0) | duplicate.any(axis=0)):
        anomalies.append({'timestamp_as_written': iso(START + int(slot)*NS), 'meters_missing': int(missing[:,slot].sum()), 'meters_duplicate': int(duplicate[:,slot].sum()), 'duplicate_extra_rows': int(np.where(duplicate[:,slot], occurrence[:,slot]-1, 0).sum())})
    date_anomalies = {}
    for item in anomalies:
        date = item['timestamp_as_written'][:10]
        agg = date_anomalies.setdefault(date, {'missing_meter_slots': 0, 'duplicate_meter_slots': 0, 'duplicate_extra_rows': 0, 'affected_clock_slots': 0})
        agg['missing_meter_slots'] += item['meters_missing']
        agg['duplicate_meter_slots'] += item['meters_duplicate']
        agg['duplicate_extra_rows'] += item['duplicate_extra_rows']
        agg['affected_clock_slots'] += 1
    gap_counts = collections.Counter()
    for row in occurrence:
        gaps, counts = np.unique(np.diff(np.flatnonzero(row)), return_counts=True)
        gap_counts.update({str(int(gap)*15): int(n) for gap,n in zip(gaps,counts)})
    coverage = {'expected_slots_per_meter_2024': SLOTS, 'meters': len(ids), 'row_count_distribution': summarize_counts(meter_rows), 'distinct_quarter_slots_distribution': summarize_counts(present_meter), 'missing_slots_distribution': summarize_counts(missing_meter), 'duplicate_keys_distribution': summarize_counts(duplicate_meter), 'duplicate_extra_rows_distribution': summarize_counts(extra_meter), 'meters_with_every_expected_slot': int((missing_meter==0).sum()), 'meters_with_exactly_one_row_every_slot': int(((missing_meter==0)&(duplicate_meter==0)).sum()), 'total_missing_meter_slots': int(missing.sum()), 'duplicate_meter_timestamp_keys': int(duplicate.sum()), 'duplicate_extra_rows': int(extra_meter.sum()), 'max_key_multiplicity': int(occurrence.max()), 'date_anomalies': date_anomalies, 'sorted_distinct_timestamp_gap_minutes_counts': dict(gap_counts)}
    out['numeric'] = numeric
    out['storage_dtypes_by_column'] = {c: sorted(vals) for c,vals in dtypes.items()}
    out['meter_count_including_missing_ID_sentinel'] = len(ids)
    out['missing_id_sentinel_present'] = '<NULL_EAN_ID>' in idmap
    out['labels'] = {c: {'row_value_counts': dict(label_counts[c]), 'meters_with_multiple_values': sum(len(m[c])>1 for m in meter_labels), 'meter_value_set_counts': dict(collections.Counter('|'.join(sorted(m[c])) for m in meter_labels))} for c in LABELS}
    out['timestamps'] = {'format_checked': 'YYYY-MM-DDTHH:MM:SS.000Z', 'min': iso(global_min), 'max': iso(global_max), 'invalid_parse_rows': ts_invalid, 'noncanonical_format_rows': ts_noncanonical, 'null_rows': ts_null, 'valid_non_quarter_hour_rows': ts_offgrid, 'valid_outside_2024_grid_rows': ts_outside, 'unusual_key_count': len(unusual_keys), 'unusual_duplicate_key_count': sum(n>1 for n in unusual_keys.values()), 'unusual_duplicate_extra_rows': sum(n-1 for n in unusual_keys.values() if n>1)}
    out['datum'] = {'min': global_date_min, 'max': global_date_max, 'invalid_or_noncanonical_rows': datum_invalid, 'timestamp_date_string_mismatch_rows': datum_mismatch}
    out['coverage'] = coverage
    out['zip_stream_completed_crc_checked'] = True
    out['elapsed_seconds'] = round(time.monotonic()-started,2)
    out['read_strategy'] = {'pandas_chunksize':250000, 'csv_columns_loaded_as_strings_before_explicit_parsing':True, 'coverage_counts_dtype':'uint32', 'key_semantics':'EAN_ID + written wall-clock timestamp fields; literal Z is preserved, but source legend documents CET/CEST. No UTC interpretation or DST correction. Synthetic array indices measure naive clock slots.'}
    out['timestamps']['timezone_semantics'] = 'Source legend documents Belgian local CET/CEST. Literal Z is retained only as source text; no UTC instant is inferred.'
    out['coverage']['coverage_interpretation'] = 'Missing and duplicate slots are relative to a naive 366-day grid. Belgian local CET/CEST expects four absent spring slots and four repeated autumn slots per complete meter-year.'
    dump(output / (stem + '.json'), out)
    dump(output / (stem + '.slot-anomalies.json'), anomalies)
    dump(output / (stem + '.internal-meters.json'), [{'id':mid, 'rows':int(meter_rows[i]), 'min_timestamp':iso(meter_ts_min[i]) if meter_ts_min[i] != np.iinfo(np.int64).max else None, 'max_timestamp':iso(meter_ts_max[i]) if meter_ts_max[i] != np.iinfo(np.int64).min else None, 'labels':{c:sorted(v) for c,v in meter_labels[i].items()}} for i,mid in enumerate(ids)])
    np.savez_compressed(output / (stem + '.coverage.npz'), ids=np.asarray(ids), occurrence=occurrence, start_ns=np.asarray(START), slot_ns=np.asarray(NS), meter_rows=meter_rows)
    print(json.dumps({'completed_file':stem,'rows':out['rows'],'meters':len(ids),'coverage':coverage,'timestamps':out['timestamps'],'elapsed_s':out['elapsed_seconds']}),flush=True)
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--archive', type=Path, required=True, help='Path to the supplied raw.zip archive.')
    parser.add_argument('--output', type=Path, required=True, help='Audit directory outside the repository; contains internal meter IDs.')
    subset = parser.add_mutually_exclusive_group()
    subset.add_argument('--part1', action='store_true', help='Audit only the four electricity Part 1 CSVs.')
    subset.add_argument('--part2', action='store_true', help='Audit only the four electricity Part 2 CSVs.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parts = [1] if args.part1 else [2] if args.part2 else [1, 2]
    prefixes = tuple(f'raw/P6269_1_50_DMK_Sample_Elek_2024_Deel_{part}/' for part in parts)
    results = []
    with zipfile.ZipFile(args.archive) as z:
        members = [i for i in z.infolist() if i.filename.startswith(prefixes) and i.filename.lower().endswith('.csv')]
        if len(members) != 4 * len(parts):
            parser.error(f'Expected {4 * len(parts)} electricity CSV members, found {len(members)}.')
        for member in members:
            results.append(audit(z, member, args.output))
    ids_by_file = {}
    for r in results:
        stem = Path(r['member']).stem
        ids_by_file[stem] = {m['id'] for m in json.loads((args.output/(stem+'.internal-meters.json')).read_text())}
    pairs = []
    names = list(ids_by_file)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            pairs.append({'left':a,'right':b,'shared_meters':len(ids_by_file[a]&ids_by_file[b])})
    dump(args.output/'summary.json',{'files':[{'member':r['member'],'rows':r['rows'],'meters':r['meter_count_including_missing_ID_sentinel']} for r in results], 'total_rows':sum(r['rows'] for r in results),'unique_meters':len(set.union(*ids_by_file.values())),'pairwise_meter_overlap':pairs})
    print(json.dumps({'all_complete':True,'summary_path':str(args.output/'summary.json')}),flush=True)


if __name__ == '__main__':
    main()
