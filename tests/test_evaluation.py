import copy
import numpy as np
import pytest
from test_training import prepared_fixture


def test_evaluation_physical_units_counts_and_permutation():
    from cgan_research.evaluation import evaluate,empirical_baseline,validation_score
    from cgan_research.metrics import fit_reference_from_prepared
    p=prepared_fixture(); p.household_ids=np.array([f'h{i//2}' for i in range(22)])
    ref=fit_reference_from_prepared(p)
    b=empirical_baseline(p,10,17)
    assert all(any(np.array_equal(x,y) for y in p.loads[p.splits['train']]) for x in b.loads)
    a=evaluate(p,b,ref,'validation',reference_replicates=2)
    assert len(a['rows'])==2*5*8
    assert all(r['exploratory'] for r in a['rows'])
    assert all(r['value'] is None for r in a['rows'] if r['metric']=='frechet')
    assert set(r['bins'] for r in a['sensitivity'])=={15,30,60}
    assert a['metadata']['unsupported_real_count']==0
    score=validation_score(a,ref)
    assert score['available'] and score['metadata']['split']=='validation'
    b.loads=b.loads[::-1]; b.labels=b.labels[::-1]
    c=evaluate(p,b,ref,'validation',reference_replicates=2)
    for left,right in zip(a['rows'],c['rows']):
        assert {k:v for k,v in left.items() if k!='value'}=={k:v for k,v in right.items() if k!='value'}
        if left['value'] is None: assert right['value'] is None
        else: assert left['value']==pytest.approx(right['value'],abs=1e-12)
    for r,s in zip(a['profile_metrics'],c['profile_metrics']):
        assert r['hourly_mean_rmse']==pytest.approx(s['hourly_mean_rmse'])
    b.representation='normalized'
    with pytest.raises(ValueError,match='physical'): evaluate(p,b,ref,'validation')
    with pytest.raises(ValueError,match='validation'):
        validation_score({**a,'metadata':{**a['metadata'],'split':'test'}},ref)


def test_profile_diagnostics_and_cluster_sampling():
    from cgan_research.evaluation import profile_diagnostics,cluster_bootstrap,aggregate_rows
    x=np.tile(np.arange(24),(3,1)).astype(float)
    d=profile_diagnostics(x,x,np.concatenate((x,np.ones((1,24)))))
    assert d['daily_energy_wasserstein']==0 and d['ramp_wasserstein']==0
    assert d['exact_training_match_rate']==1 and d['generated_diversity_rms']==0
    assert d['generated_autocorrelation'][0]==pytest.approx(0.875)
    flat=profile_diagnostics(np.ones((2,24)),np.ones((2,24)),np.ones((2,24)))
    assert flat['real_autocorrelation']==[None]*6
    assert flat['real_constant_profiles']==2
    boot=cluster_bootstrap(np.array([0.,0.,10.]),np.array(['a','a','b']),np.mean,resamples=100,seed=42)
    assert boot['n_groups']==2 and boot['estimate']==pytest.approx(10/3)
    assert boot['interval_95']==[0.,10.]
    rows=[{'metric':'wasserstein','feature':'near_peak','value':1.,'n_real':9},{'metric':'wasserstein','feature':'near_peak','value':3.,'n_real':1},{'metric':'wasserstein','feature':'near_peak','value':None,'n_real':2}]
    s=aggregate_rows(rows)[0]
    assert s['macro']==2 and s['real_count_weighted']==pytest.approx(1.2) and s['unavailable_cells']==1


def test_public_metadata_excludes_meter_ids_and_unsupported_are_counted():
    from cgan_research.evaluation import evaluate,empirical_baseline,validation_score
    from cgan_research.metrics import fit_reference_from_prepared
    p=prepared_fixture()
    p.household_ids=np.array([f'private-meter-{i}' for i in range(22)])
    p.manifest['provenance']['meter_metadata']={'PRIVATE-123':{'EV':'0'}}
    p.labels[18]=-1
    p.conditions=np.array(['winter|weekday']*22); p.conditions[18]='summer|weekday'
    r=evaluate(p,empirical_baseline(p,5),fit_reference_from_prepared(p),reference_replicates=0)
    import json
    public={k:v for k,v in r.items() if not k.startswith('_')}
    assert 'PRIVATE-123' not in json.dumps(public)
    cells=[v for v in r['rows'] if v['condition']=='summer|weekday']
    assert len(cells)==40 and all(c['value'] is None for c in cells)
    assert all(c['excluded_real_count']>=1 for c in r['summary'])
    ref={'constant':[True]*5,'scale':[1.]*5}
    assert not validation_score(r,ref)['available']


def test_generation_from_different_split_is_rejected():
    from cgan_research.evaluation import evaluate,empirical_baseline
    from cgan_research.metrics import fit_reference_from_prepared
    p=prepared_fixture(); b=empirical_baseline(p,5)
    p.manifest['split_fingerprint']='a-different-partition'
    p.splits['validation']=p.splits['train'][:2]
    with pytest.raises(ValueError,match='split|fingerprint'):
        evaluate(p,b,fit_reference_from_prepared(p),reference_replicates=0)
