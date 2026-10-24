from dataclasses import replace
import json
import numpy as np
import pytest
from cgan_research.types import DailyProfiles
from cgan_research.prepare import prepare_dataset,load_prepared,transform_loads,inverse_loads,encode_conditions


def sample():
    n=40
    return DailyProfiles(profile_ids=np.asarray([f'p{i}' for i in range(n)]),household_ids=np.asarray([f'm{i//2:02}' for i in range(n)]),dates=np.asarray(['2024-01-01','2024-01-02']*20),loads=np.tile(np.linspace(-3,9,24),(n,1)),conditions=np.asarray(['winter|weekday']*n),units='kWh',source_fingerprint='synthetic-fixture',metadata={'synthetic':True})


def test_split_train_only_and_roundtrip(tmp_path):
    d=sample(); a=prepare_dataset(d,{},tmp_path/'a'); b=prepare_dataset(d,{},tmp_path/'b')
    assert [len(a.splits[k]) for k in ('train','validation','test')]==[28,6,6]
    groups=[set(a.household_ids[a.splits[k]]) for k in a.splits]
    assert not(groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    assert a.manifest['split_fingerprint']==b.manifest['split_fingerprint']
    loads=d.loads.copy(); loads[a.splits['test']]=1000
    cond=d.conditions.copy(); cond[a.splits['test']]='summer|weekend'
    c=prepare_dataset(replace(d,loads=loads,conditions=cond),{},tmp_path/'c')
    assert c.manifest['normalizer']==a.manifest['normalizer']
    assert c.normalized[c.splits['test']].max()>1 and (c.labels[c.splits['test']]==-1).all()
    assert c.normalized.dtype==np.float32 and c.loads.dtype==np.float64 and c.labels.dtype==np.int64
    np.testing.assert_allclose(inverse_loads(c.normalized,c.manifest['normalizer']),c.loads,atol=1e-4)
    loaded=load_prepared(tmp_path/'c'); np.testing.assert_array_equal(loaded.loads,c.loads)
    assert loaded.manifest['synthetic'] is True and loaded.manifest['representations']['loads']=='physical'
    with pytest.raises(FileExistsError): prepare_dataset(d,{},tmp_path/'a')


def test_constant_and_explicit_protocol_errors(tmp_path):
    d=sample(); a=prepare_dataset(replace(d,loads=np.full((40,24),-5.)),{},tmp_path/'const')
    assert a.manifest['normalizer']=={'min':-5.,'scale':1.,'constant':True}
    assert np.all(a.normalized==0)
    for field,match in [('household_ids','household'),('dates','dates')]:
        with pytest.raises(ValueError,match=match):
            prepare_dataset(replace(d,**{field:np.asarray(['']*40)}),{},tmp_path/field)
    duplicate=d.profile_ids.copy(); duplicate[1]=duplicate[0]
    with pytest.raises(ValueError,match='duplicate'): prepare_dataset(replace(d,profile_ids=duplicate),{},tmp_path/'dup')
    ids=d.household_ids.copy(); ids[1]=ids[0]; dates=d.dates.copy(); dates[1]=dates[0]
    with pytest.raises(ValueError,match='duplicate'): prepare_dataset(replace(d,dates=dates),{},tmp_path/'dupkey')


def test_npz_tamper_detected_and_no_pickle(tmp_path):
    p=tmp_path/'p'; prepare_dataset(sample(),{},p)
    with np.load(p/'arrays.npz',allow_pickle=False) as a: assert a['profile_ids'].dtype.kind=='U'
    with (p/'arrays.npz').open('ab') as f: f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash|fingerprint'): load_prepared(p)


def test_condition_encoder_reports_unknown():
    labels,supported=encode_conditions(np.asarray(['winter|weekday','new']),{'winter|weekday':0})
    assert labels.tolist()==[0,-1] and supported.tolist()==[True,False]
