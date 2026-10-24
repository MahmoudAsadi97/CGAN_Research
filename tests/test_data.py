import csv
import zipfile
from dataclasses import replace
import numpy as np
import pytest
from cgan_research.types import DatasetConfig
from cgan_research.data import read_daily_csv, read_electricity_archive


def cfg(**kwargs):
    return DatasetConfig(hourly_columns=tuple(f'load_{i:02}' for i in range(24)), profile_id_column='id', household_id_column='meter', date_column='date', timezone='Europe/Brussels', units='kWh', interval_convention='start', season_months={'winter': (12,1,2), 'spring': (3,4,5), 'summer': (6,7,8), 'autumn': (9,10,11)}, weekend_days=(5,6), **kwargs)


def wide(tmp_path, rows):
    p=tmp_path/'daily.csv'
    with p.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['id','meter','date']+[f'load_{i:02}' for i in range(24)]+['EV','PV']); w.writerows(rows)
    return p


def test_wide_negative_flat_calendar_and_nonfinite(tmp_path):
    p=wide(tmp_path,[['a','x','2024-01-01']+[-2]*24+['1','0'],['b','y','2024-06-01']+[3]*24+['0','1'],['c','z','2024-02-01']+['nan']+[0]*23+['0','0']])
    d,a=read_daily_csv(p,cfg())
    assert d.loads.dtype==np.float64 and d.loads.shape==(2,24)
    assert d.loads[0,0]==-2
    assert d.conditions.tolist()==['winter|weekday','summer|weekend']
    assert a.accepted==2 and a.reasons['nonfinite_reading']==1
    d,_=read_daily_csv(p,replace(cfg(),holiday_dates=('2024-01-01',)))
    assert d.conditions[0]=='winter|weekend'


def test_missing_fields_explicit_and_duplicate_conflict(tmp_path):
    p=wide(tmp_path,[['a','x','2024-01-01']+[1]*24+['0','0'],['b','x','2024-01-01']+[2]*24+['0','0']])
    d,a=read_daily_csv(p,cfg()); assert len(d.loads)==0 and a.reasons['duplicate_group_date']==2
    d,_=read_daily_csv(p,replace(cfg(),date_column=None,household_id_column=None))
    assert d.dates.tolist()==['',''] and d.household_ids.tolist()==['',''] and d.conditions.tolist()==['','']
    with pytest.raises(ValueError,match='condition'):
        replace(cfg(),condition_columns=('EV',))


def test_both_dst_days_excluded_even_24_columns(tmp_path):
    p=wide(tmp_path,[['a','x','2024-03-31']+[1]*24+['0','0'],['b','y','2024-10-27']+[1]*24+['0','0']])
    d,a=read_daily_csv(p,cfg()); assert len(d.loads)==0
    assert a.reasons['dst_transition']==2 and a.excluded_by_month=={'03':1,'10':1}
    with pytest.raises(ValueError,match='timezone'):
        replace(cfg(),timezone='+01:00')


RAW=['EAN_ID','Datum','Datum_Startuur','Volume_Afname_KWh','Volume_Injectie_KWh','Warmtepomp_Indicator','Elektrisch_Voertuig_Indicator','PV_Installatie_Indicator','Contract_Categorie']

def raw_archive(tmp_path, bad=False, duplicate=False):
    import io
    s=io.StringIO(); w=csv.writer(s); w.writerow(RAW)
    for day in ('2024-01-01','2024-01-02','2024-03-31','2024-10-27'):
        for q in range(96):
            if day=='2024-01-02' and q==95: continue
            row=['meter-private',day,f'{day}T{q//4:02}:{15*(q%4):02}:00.000Z', 'nan' if bad and q==5 else '0.25','0.5','0','1','1','Residentieel']
            w.writerow(row)
            if duplicate and day=='2024-01-01' and q==0: w.writerow(row)
    p=tmp_path/'raw.zip'
    with zipfile.ZipFile(p,'w') as z: z.writestr('raw/P6269_1_50_DMK_Sample_Elek_2024_Deel_1/fixture.csv',s.getvalue())
    return p


def test_raw_sums_energy_preserves_net_export_and_detection_semantics(tmp_path):
    d,a=read_electricity_archive(raw_archive(tmp_path),cfg())
    assert d.loads.shape==(1,24) and np.all(d.loads==-1)
    assert d.conditions.tolist()==['winter|weekday']
    assert a.reasons['dst_transition']==2 and a.reasons['incomplete_day']>=1
    assert d.metadata['grouping_semantics']=='meter proxy; independently verified household mapping unavailable'
    assert 'not detected does not establish absence' in d.metadata['ev_detection_semantics']
    assert d.metadata['meter_metadata']['meter-private']['Elektrisch_Voertuig_Indicator']=='1'
    assert 'meter-private' not in str(a.to_dict())


@pytest.mark.parametrize('kwargs,reason',[({'bad':True},'invalid_interval_energy'),({'duplicate':True},'duplicate_interval')])
def test_raw_invalid_and_duplicate_day_quarantined(tmp_path,kwargs,reason):
    d,a=read_electricity_archive(raw_archive(tmp_path,**kwargs),cfg())
    assert len(d.loads)==0 and a.reasons[reason]>=1


def test_yaml_configuration_and_chunked_adapter(tmp_path):
    import yaml
    config=yaml.safe_load(open('configs/dataset_electricity.yaml'))
    config['dataset']['chunksize']=31
    config['dataset']['expected_cohorts']=1
    d,a=read_electricity_archive(raw_archive(tmp_path),config)
    assert d.loads.shape==(1,24) and a.metadata['raw_rows']==383


def test_bad_raw_timestamp_has_explicit_reason(tmp_path):
    p=raw_archive(tmp_path)
    with zipfile.ZipFile(p) as z:
        name=z.namelist()[0]; text=z.read(name).decode()
    text=text.replace('2024-01-01T00:00:00.000Z','2024-01-01T00:01:00.000Z')
    with zipfile.ZipFile(p,'w') as z: z.writestr(name,text)
    d,a=read_electricity_archive(p,cfg())
    assert len(d.loads)==0 and a.reasons['invalid_interval_timestamp']==1


def test_audited_electricity_cohorts_at_archive_root(tmp_path):
    import yaml
    p=raw_archive(tmp_path)
    with zipfile.ZipFile(p) as z: name=z.namelist()[0]; content=z.read(name)
    with zipfile.ZipFile(p,'w') as z: z.writestr(name.removeprefix('raw/'),content)
    config=yaml.safe_load(open('configs/dataset_electricity.yaml'))
    config['dataset']['expected_cohorts']=1
    d,a=read_electricity_archive(p,config)
    assert d.loads.shape==(1,24) and a.metadata['cohorts']==1


def test_release_config_requires_all_eight_cohorts(tmp_path):
    import yaml
    config=yaml.safe_load(open('configs/dataset_electricity.yaml'))
    with pytest.raises(ValueError,match='cohort'):
        read_electricity_archive(raw_archive(tmp_path),config)


def test_local_variant_records_actual_and_published_provenance(tmp_path):
    import yaml
    config=yaml.safe_load(open('configs/dataset_electricity.yaml'))
    config['dataset']['expected_cohorts']=1
    p=raw_archive(tmp_path)
    d,_=read_electricity_archive(p,config)
    assert d.metadata['source_integrity']=='local_variant_unverified_release'
    assert d.metadata['source_archive_bytes']==p.stat().st_size
    assert d.metadata['published_release_sha256']==config['source']['sha256']
    assert d.metadata['published_release_bytes']==395824014
    assert d.metadata['source_archive_sha256']!=d.metadata['published_release_sha256']
