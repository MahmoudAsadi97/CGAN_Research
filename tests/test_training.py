from types import SimpleNamespace
import json
import numpy as np
import pytest
import torch


def prepared_fixture():
    x = np.random.default_rng(8).normal(size=(22,24))
    return SimpleNamespace(loads=x * 3 - 2, normalized=x.astype('float32'), labels=np.arange(22)%2,
        splits={'train':np.arange(18), 'validation':np.arange(18,20), 'test':np.arange(20,22)},
        manifest={'data_fingerprint':'synthetic-data', 'split_fingerprint':'synthetic-split',
          'normalizer':{'min':-2.,'scale':3.,'constant':False},'condition_mapping':{'winter|weekday':0,'winter|weekend':1},
          'units':'kWh','provenance':{'synthetic':True}})


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): assert_tree_equal(x,y)
    else: assert a==b


@pytest.mark.parametrize('sampling',['fixed','shuffle','balanced'])
def test_exact_mid_epoch_resume_across_validation(tmp_path,sampling):
    from cgan_research.training import train, TrainConfig
    from cgan_research.checkpoint import load_checkpoint
    p=prepared_fixture()
    cfg=TrainConfig(device='cpu',max_steps=6,max_epochs=4,batch_size=5,seed=17,sampling=sampling,validation_interval=2,checkpoint_interval=1,samples_per_condition=7)
    def callback(g,step):
        # A callback must never change training RNG streams.
        torch.rand(3); np.random.rand(3)
        return {'score':1/step,'available':True,'metadata':{'split':'validation'}}
    train(p,cfg,tmp_path/'full',validation_callback=callback)
    train(p,cfg,tmp_path/'part',validation_callback=callback,stop_after_steps=3)
    train(p,cfg,tmp_path/'part',resume=tmp_path/'part'/'last.pt',validation_callback=callback)
    a,b=(load_checkpoint(tmp_path/n/'last.pt') for n in ('full','part'))
    for key in ('generator','discriminator','g_optimizer','d_optimizer','step','epoch','cursor','order','torch_rng','numpy_rng','python_rng','history','best_score','best_step','next_validation_step'):
        assert_tree_equal(a[key],b[key])
    assert a['step']==6 and a['best_step']==6
    assert json.loads((tmp_path/'part'/'status.json').read_text())['status']=='completed'
    assert (tmp_path/'part'/'best.pt').is_file()
    with pytest.raises(ValueError,match='configuration'):
        train(p,TrainConfig(device='cpu',max_steps=7),tmp_path/'part',resume=tmp_path/'part'/'last.pt')


def test_updates_inverse_units_and_no_training_rng_mutation(tmp_path):
    from cgan_research.training import train, generate, TrainConfig
    from cgan_research.models import Generator,Discriminator
    from cgan_research.checkpoint import load_checkpoint
    cfg=TrainConfig(device='cpu',max_steps=2,max_epochs=2,batch_size=7,validation_interval=0)
    torch.manual_seed(cfg.seed)
    g0,d0=Generator(2),Discriminator(2)
    train(prepared_fixture(),cfg,tmp_path/'run')
    s=load_checkpoint(tmp_path/'run'/'last.pt')
    assert any(not torch.equal(v,s['generator'][k]) for k,v in g0.state_dict().items())
    assert any(not torch.equal(v,s['discriminator'][k]) for k,v in d0.state_dict().items())
    torch.manual_seed(91); before=torch.get_rng_state()
    b=generate(tmp_path/'run'/'last.pt',[0,1],9,31)
    assert torch.equal(before,torch.get_rng_state())
    g=Generator(2); g.load_state_dict(s['generator']); g.eval()
    z=torch.randn(18,24,generator=torch.Generator().manual_seed(31))
    with torch.no_grad(): expected=g(z,torch.tensor([0]*9+[1]*9)).numpy().astype('float64')*3-2
    np.testing.assert_allclose(b.loads,expected,atol=1e-6)
    assert b.units=='kWh' and b.representation=='physical' and b.loads.dtype==np.float64
    assert np.isfinite([r['g_loss'] for r in s['history']]).all()
    with pytest.raises(ValueError,match='condition'):
        generate(tmp_path/'run'/'last.pt',[2],9,31)
    with pytest.raises(FileExistsError):
        train(prepared_fixture(),cfg,tmp_path/'run')


def test_failed_device_persists_status(tmp_path,monkeypatch):
    from cgan_research.training import train,TrainConfig
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA'):
        train(prepared_fixture(),TrainConfig(device='cuda',max_steps=2),tmp_path/'failed')
    assert json.loads((tmp_path/'failed'/'status.json').read_text())['status']=='failed'


def test_resume_repairs_interrupted_best_publication(tmp_path,monkeypatch):
    import cgan_research.training as training
    from cgan_research.checkpoint import load_checkpoint
    cfg=training.TrainConfig(device='cpu',max_steps=3,batch_size=5,validation_interval=1,checkpoint_interval=1)
    callback=lambda g,s: {'score':1.,'available':True,'metadata':{'split':'validation'}}
    original=training.save_checkpoint
    def fail_best(path,state):
        if path.name=='best.pt': raise OSError('interrupted best publication')
        return original(path,state)
    monkeypatch.setattr(training,'save_checkpoint',fail_best)
    with pytest.raises(OSError):
        training.train(prepared_fixture(),cfg,tmp_path/'run',validation_callback=callback)
    monkeypatch.setattr(training,'save_checkpoint',original)
    training.train(prepared_fixture(),cfg,tmp_path/'run',resume=tmp_path/'run'/'last.pt',validation_callback=callback)
    best=load_checkpoint(tmp_path/'run'/'best.pt')
    assert best['step']==1
    assert load_checkpoint(tmp_path/'run'/'last.pt')['best_step']==1


def test_cuda_workspace_preflight(monkeypatch):
    from cgan_research.training import configure_cuda
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    configure_cuda()
    import os
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8',':16:8')
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG','invalid')
    with pytest.raises(ValueError,match='CUBLAS_WORKSPACE_CONFIG'): configure_cuda()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.])
def test_invalid_weight_decay_fails_before_run_creation(value):
    from cgan_research.training import TrainConfig
    with pytest.raises(ValueError): TrainConfig(weight_decay=value)


def test_evaluation_keeps_training_seed_separate_from_sampling_seed(tmp_path):
    from cgan_research.training import train,generate,TrainConfig
    from cgan_research.evaluation import evaluate
    from cgan_research.metrics import fit_reference_from_prepared
    p=prepared_fixture()
    train(p,TrainConfig(device='cpu',seed=17,max_steps=1,validation_interval=0),tmp_path/'run')
    b=generate(tmp_path/'run'/'last.pt',[0,1],5,2025)
    r=evaluate(p,b,fit_reference_from_prepared(p),reference_replicates=0)
    assert r['metadata']['seed']==17
    assert r['metadata']['generation_seed']==2025


def test_environment_records_source_and_dependency_versions():
    from cgan_research.training import environment
    report=environment('cpu')
    assert len(report['source_sha256'])==64
    assert {'torch','numpy','pandas','scipy','matplotlib','PyYAML'} <= set(report['packages'])
