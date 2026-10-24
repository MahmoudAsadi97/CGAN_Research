import copy
import json
import numpy as np
import pytest
from test_training import prepared_fixture


def test_paired_seed_and_household_cluster_comparison(tmp_path):
    from cgan_research.evaluation import evaluate,empirical_baseline,save_evaluation,compare_runs
    from cgan_research.metrics import fit_reference_from_prepared
    p=prepared_fixture()
    # Two meters each contribute both conditions; resample entire meter blocks.
    p.splits['validation']=np.arange(14,22)
    p.splits['train']=np.arange(14)
    p.household_ids=np.array([f'private-meter-{i//2}' for i in range(22)])
    ref=fit_reference_from_prepared(p)
    dirs=[]
    for model,offset in [('reference',0.),('candidate',2.)]:
        for seed in (17,42):
            batch=empirical_baseline(p,10,seed); batch.loads+=offset; batch.model=model
            result=evaluate(p,batch,ref,reference_replicates=0)
            directory=tmp_path/f'{model}-{seed}'; save_evaluation(result,directory); dirs.append(directory)
            assert 'private-meter' not in (directory/'evaluation.json').read_text()
            assert (directory/'private_records.npz').exists()
    report=compare_runs(dirs,resamples=8)
    assert report['paired_seeds']==[17,42]
    assert report['household_intervals']['n_groups']==4
    assert report['household_intervals']['available_replicates']==8
    assert report['household_intervals']['interval_95'] is not None
    assert report==compare_runs(dirs,resamples=8)
    with pytest.raises(ValueError,match='Duplicate'): compare_runs(dirs+[dirs[0]])


def test_public_comparison_preflight_rejects_incompatible_sampling_and_references(tmp_path):
    from cgan_research.evaluation import evaluate,empirical_baseline,save_evaluation,compare_runs
    from cgan_research.metrics import fit_reference_from_prepared
    p=prepared_fixture(); ref=fit_reference_from_prepared(p)
    dirs=[]
    for model in ('reference','candidate'):
        b=empirical_baseline(p,5,42,training_seed=17); b.model=model
        r=evaluate(p,b,ref,reference_replicates=0)
        d=tmp_path/model; save_evaluation(r,d); (d/'private_records.npz').unlink(); dirs.append(d)
    path=dirs[1]/'evaluation.json'; original=json.loads(path.read_text())
    for mutation in ('seed','reference','count'):
        modified=copy.deepcopy(original)
        if mutation=='seed': modified['metadata']['generation_seed']=99
        elif mutation=='reference': modified['reference']['scale'][0]*=10
        else:
            for row in modified['rows']: row['n_generated']+=1
        path.write_text(json.dumps(modified))
        with pytest.raises(ValueError,match='sampling|reference|count'):
            compare_runs(dirs,resamples=2)
