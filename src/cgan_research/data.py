"""Validated daily CSV intake and audited Belgian 2024 quarter-hour adapter.

The release's written Z is deliberately treated as a literal: its legend and
observed clock changes establish Belgian wall time, not UTC instants.
"""
import calendar
from collections import Counter
from dataclasses import asdict
from datetime import date,datetime,time,timedelta,timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zipfile
import numpy as np
import pandas as pd
from .types import DatasetConfig,DailyProfiles,AuditReport
from .download import sha256_file,validate_zip_members

RAW_COLUMNS=['EAN_ID','Datum','Datum_Startuur','Volume_Afname_KWh','Volume_Injectie_KWh','Warmtepomp_Indicator','Elektrisch_Voertuig_Indicator','PV_Installatie_Indicator','Contract_Categorie']
METADATA_COLUMNS=RAW_COLUMNS[5:]


def dataset_config(config):
    if isinstance(config,DatasetConfig): return config
    values=dict(config)
    # Dataset YAML contains provenance alongside the explicit parser configuration.
    return DatasetConfig(**values.get('dataset',values))


def condition_for_date(day,config):
    season=next(name for name,months in config.season_months.items() if day.month in months)
    daytype='weekend' if day.weekday() in config.weekend_days or day.isoformat() in config.holiday_dates else 'weekday'
    return f'{season}|{daytype}'


def is_dst_day(day,zone):
    tz=ZoneInfo(zone)
    start=datetime.combine(day,time(),tzinfo=tz).astimezone(timezone.utc)
    end=datetime.combine(day+timedelta(days=1),time(),tzinfo=tz).astimezone(timezone.utc)
    return (end-start)!=timedelta(hours=24)


def _exclude(report,reason,day,config,n=1):
    report.quarantined+=int(n); report.reasons[reason]=report.reasons.get(reason,0)+int(n)
    if day is not None:
        month=f'{day.month:02}'; season=condition_for_date(day,config).split('|')[0]
        report.excluded_by_month[month]=report.excluded_by_month.get(month,0)+int(n)
        report.excluded_by_season[season]=report.excluded_by_season.get(season,0)+int(n)


def _fingerprint(path,config):
    return hashlib.sha256((sha256_file(path)+json.dumps(asdict(config),sort_keys=True)).encode()).hexdigest()


def read_daily_csv(path,config):
    config=dataset_config(config); path=Path(path)
    if path.suffix.lower()!='.csv': raise ValueError('Unsupported input type; daily-wide intake requires a mapped .csv file')
    frame=pd.read_csv(path,dtype=str,keep_default_na=False)
    required=[config.profile_id_column,*config.hourly_columns]
    required += [x for x in (config.household_id_column,config.date_column) if x]
    if set(required)-set(frame.columns): raise ValueError('Missing configured columns: '+', '.join(sorted(set(required)-set(frame.columns))))
    report=AuditReport(); accepted=[]
    ids=frame[config.profile_id_column]
    duplicate=ids.duplicated(keep=False).to_numpy()
    if config.household_id_column and config.date_column:
        duplicate |= frame.duplicated([config.household_id_column,config.date_column],keep=False).to_numpy()
    values=frame[list(config.hourly_columns)].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
    days=[]; groups=[]; conditions=[]
    for i,row in enumerate(frame.to_dict('records')):
        day=None
        if config.date_column:
            try: day=date.fromisoformat(row[config.date_column])
            except ValueError:
                _exclude(report,'invalid_date',None,config); continue
        if duplicate[i]: _exclude(report,'duplicate_group_date',day,config); continue
        if not row[config.profile_id_column]: _exclude(report,'missing_profile_id',day,config); continue
        if not np.isfinite(values[i]).all(): _exclude(report,'nonfinite_reading',day,config); continue
        if day is not None and is_dst_day(day,config.timezone): _exclude(report,'dst_transition',day,config); continue
        accepted.append(i); days.append(day.isoformat() if day else '')
        groups.append(row[config.household_id_column] if config.household_id_column else '')
        conditions.append(condition_for_date(day,config) if day else '')
    report.accepted=len(accepted)
    metadata={'synthetic':config.synthetic,'timezone':config.timezone,'interval_convention':config.interval_convention,'calendar':{'season_months':config.season_months,'weekend_days':config.weekend_days,'holiday_dates':config.holiday_dates},'raw_metadata_fields_not_conditions':[c for c in frame if c not in required]}
    return DailyProfiles(ids.iloc[accepted].to_numpy(),np.asarray(groups),np.asarray(days),values[accepted],np.asarray(conditions),config.units,_fingerprint(path,config),metadata),report


def read_electricity_archive(path,config):
    """Stream each cohort into at most max_meters × year_days × 96 cells.

    All profiles with any missing/repeated/invalid interval are quarantined;
    transition dates are excluded independently of their observed row counts.
    Entirely missing days count as incomplete within each observed meter-year.
    """
    source_config=dict(config).get('source',{}) if isinstance(config,dict) else {}
    config=dataset_config(config)
    if config.units!='kWh' or config.interval_convention!='start' or config.timezone!='Europe/Brussels':
        raise ValueError('Release adapter requires kWh interval-start energy in Europe/Brussels local time')
    path=Path(path)
    if path.suffix.lower()!='.zip': raise ValueError('Electricity release adapter requires the dataset .zip archive')
    ndays=366 if calendar.isleap(config.source_year) else 365
    first=date(config.source_year,1,1); dates=[first+timedelta(days=i) for i in range(ndays)]
    transition=np.asarray([is_dst_day(d,config.timezone) for d in dates])
    start_ns=pd.Timestamp(first).value; slot_ns=900_000_000_000
    all_ids=[]; all_groups=[]; all_days=[]; all_loads=[]; all_conditions=[]
    seen_meters=set(); meter_metadata={}; report=AuditReport(metadata={'raw_rows':0,'cohorts':0,'grouping_unit':'meter proxy','weather_used':False})
    with zipfile.ZipFile(path) as z:
        infos=validate_zip_members(z,config.max_expanded_bytes)
        members=[i for i in infos if i.filename.startswith(tuple(config.electricity_prefixes)) and i.filename.lower().endswith('.csv')]
        if not members: raise ValueError('No audited electricity cohort CSV members found in release archive')
        if config.expected_cohorts is not None and len(members)!=config.expected_cohorts:
            raise ValueError(f'Expected {config.expected_cohorts} electricity cohorts, found {len(members)}')
        for info in sorted(members,key=lambda i:i.filename):
            report.metadata['cohorts']+=1
            shape=(config.max_meters_per_cohort,ndays*96)
            counts=np.zeros(shape,dtype=np.uint32); net=np.zeros(shape,dtype=np.float64)
            bad=np.zeros((shape[0],ndays),dtype=bool); bad_time=bad.copy(); meter_bad=np.zeros(shape[0],dtype=bool)
            idmap={}; ids=[]; metadata_values={}
            with z.open(info) as raw:
                for chunk in pd.read_csv(raw,dtype='string',keep_default_na=False,chunksize=config.chunksize):
                    if chunk.columns.tolist()!=RAW_COLUMNS: raise ValueError('Raw electricity schema differs from audited nine-column schema')
                    report.metadata['raw_rows']+=len(chunk)
                    raw_ids=chunk['EAN_ID']
                    if (raw_ids=='').any(): raise ValueError('Missing meter identifier; cannot safely group raw readings')
                    local_codes,local_ids=pd.factorize(raw_ids,sort=False)
                    for mid in local_ids:
                        mid=str(mid)
                        if mid not in idmap:
                            if mid in seen_meters: raise ValueError('Meter identifiers overlap cohort files; refusing duplicate measurements')
                            if len(ids)>=config.max_meters_per_cohort: raise ValueError('Cohort exceeds configured bounded meter grid')
                            idmap[mid]=len(ids); ids.append(mid)
                    codes=np.asarray([idmap[str(mid)] for mid in local_ids],dtype=np.int32)[local_codes]
                    # Keep metadata internal, validating each meter has one documented value.
                    label_frame=chunk[['EAN_ID',*METADATA_COLUMNS]].drop_duplicates()
                    for row in label_frame.itertuples(index=False,name=None):
                        mid=str(row[0]); vals=tuple(str(v) for v in row[1:])
                        if mid in metadata_values and metadata_values[mid]!=vals: meter_bad[idmap[mid]]=True
                        metadata_values[mid]=vals
                        if any(v not in ('0','1') for v in vals[:3]) or not vals[3]: meter_bad[idmap[mid]]=True
                    rawdate=chunk['Datum']; rawts=chunk['Datum_Startuur']
                    parsed=pd.to_datetime(rawts,format='%Y-%m-%dT%H:%M:%S.%fZ',errors='coerce')
                    ns=parsed.to_numpy(dtype='datetime64[ns]').view(np.int64)
                    good_time=(ns!=np.iinfo(np.int64).min)
                    canonical=rawts.str.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00\.000Z').to_numpy(dtype=bool)
                    slots=np.zeros(len(chunk),dtype=np.int64); slots[good_time]=(ns[good_time]-start_ns)//slot_ns
                    good_time &= canonical & (slots>=0)&(slots<ndays*96)
                    aligned=np.zeros(len(chunk),dtype=bool)
                    aligned[good_time]=(ns[good_time]-start_ns)%slot_ns==0
                    good_time &= aligned
                    valid_date=pd.to_datetime(rawdate,format='%Y-%m-%d',errors='coerce')
                    day_ns=valid_date.to_numpy(dtype='datetime64[ns]').view(np.int64)
                    day_index=np.zeros(len(chunk),dtype=np.int64)
                    valid_day=(day_ns!=np.iinfo(np.int64).min)
                    day_index[valid_day]=(day_ns[valid_day]-start_ns)//(86400*1_000_000_000)
                    valid_day &= (day_index>=0)&(day_index<ndays)&rawdate.str.fullmatch(r'\d{4}-\d{2}-\d{2}').to_numpy(dtype=bool)
                    date_match=(rawdate==rawts.str.slice(0,10)).to_numpy(dtype=bool)
                    imports=pd.to_numeric(chunk[RAW_COLUMNS[3]],errors='coerce').to_numpy(dtype=np.float64)
                    exports=pd.to_numeric(chunk[RAW_COLUMNS[4]],errors='coerce').to_numpy(dtype=np.float64)
                    good_energy=np.isfinite(imports)&np.isfinite(exports)&(imports>=0)&(exports>=0)
                    valid=good_time&valid_day&date_match&good_energy
                    invalid=~valid
                    # Bad dates cannot be assigned safely to a day: quarantine that meter-year.
                    meter_bad[codes[invalid&~valid_day]]=True
                    known_bad=invalid&valid_day
                    bad[codes[known_bad&~good_energy],day_index[known_bad&~good_energy]]=True
                    timestamp_bad=known_bad&(~good_time|~date_match)
                    bad_time[codes[timestamp_bad],day_index[timestamp_bad]]=True
                    if valid.any():
                        np.add.at(counts,(codes[valid],slots[valid]),1)
                        np.add.at(net,(codes[valid],slots[valid]),imports[valid]-exports[valid])
                if raw.read(1): raise ValueError('CSV stream was not exhausted')
            for mid,index in idmap.items():
                meter_metadata[mid]=dict(zip(METADATA_COLUMNS,metadata_values[mid]))
                c=counts[index].reshape(ndays,96); v=net[index].reshape(ndays,96)
                for j,day in enumerate(dates):
                    if transition[j]: reason='dst_transition'
                    elif meter_bad[index]: reason='invalid_meter_metadata_or_date'
                    elif bad_time[index,j]: reason='invalid_interval_timestamp'
                    elif bad[index,j]: reason='invalid_interval_energy'
                    elif (c[j]>1).any(): reason='duplicate_interval'
                    elif (c[j]!=1).any(): reason='incomplete_day'
                    else: reason=None
                    if reason:
                        _exclude(report,reason,day,config); continue
                    all_ids.append(hashlib.sha256((mid+'|'+day.isoformat()).encode()).hexdigest())
                    all_groups.append(mid); all_days.append(day.isoformat())
                    all_loads.append(v[j].reshape(24,4).sum(axis=1)); all_conditions.append(condition_for_date(day,config))
            seen_meters.update(ids)
            del counts,net,bad,bad_time
    report.accepted=len(all_ids)
    metadata={'synthetic':config.synthetic,'timezone':config.timezone,'timestamp_semantics':'Written Z retained as literal; source legend establishes Belgian local CET/CEST; no UTC instant inferred','units':'kWh','quantity':'net grid exchange = imported minus exported interval energy, four quarter-hours summed per hour','interval_convention':'start','grouping_semantics':'meter proxy; independently verified household mapping unavailable','ev_detection_semantics':'1 means detected BEV/PHEV; 0 means not detected; not detected does not establish absence; no verified ownership or charging-event labels','meter_metadata':meter_metadata,'heat_pump_documentation_caveat':'Source boolean descriptions contain apparent EV copy-paste text; preserve supplied flag','pv_caveat':'Positive export with PV=0 exists; flags/readings preserved','weather_used':False,'source_archive_sha256':sha256_file(path),'source_members':[i.filename for i in members],'calendar':{'season_months':config.season_months,'weekend_days':config.weekend_days,'holiday_dates':config.holiday_dates}}
    metadata['source_archive_bytes']=path.stat().st_size
    metadata['published_release_sha256']=source_config.get('sha256')
    metadata['published_release_bytes']=source_config.get('bytes')
    verified=bool(source_config.get('sha256')) and metadata['source_archive_sha256']==source_config.get('sha256') and (source_config.get('bytes') is None or metadata['source_archive_bytes']==source_config['bytes'])
    metadata['source_integrity']='verified_published_release' if verified else 'local_variant_unverified_release'
    loads=np.asarray(all_loads,dtype=np.float64).reshape(-1,24)
    return DailyProfiles(np.asarray(all_ids),np.asarray(all_groups),np.asarray(all_days),loads,np.asarray(all_conditions),'kWh',_fingerprint(path,config),metadata),report
