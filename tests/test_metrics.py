import numpy as np
import pytest


def test_distribution_identity_permutation_and_unequal_counts():
    from cgan_research.metrics import fit_metric_reference,compare_distributions
    x=np.array([[0,0,1,2,3],[1,1,2,3,4],[3,2,5,6,7]],dtype=float)
    ref=fit_metric_reference(x)
    r=compare_distributions(x,x[::-1],ref)
    for f in r['features'].values():
        for metric in ('kl','js','wasserstein','mean_difference','std_difference','frechet'):
            assert f[metric]==pytest.approx(0,abs=1e-12)
    assert r['joint_frechet']==pytest.approx(0,abs=1e-8)
    r=compare_distributions(x,np.repeat(x,2,axis=0),ref)
    assert r['features']['near_peak']['wasserstein']==0
    r=compare_distributions(x,x[:1],ref)
    assert r['features']['near_peak']['std_difference'] is None
    assert r['joint_frechet'] is None
    r=compare_distributions(x,np.empty((0,5)),ref)
    assert r['features']['near_peak']['kl'] is None


def test_histogram_tails_smoothing_and_frozen_reference():
    from cgan_research.metrics import fit_metric_reference,histogram_probabilities,compare_distributions
    ref=fit_metric_reference(np.ones((4,5)))
    old=repr(ref)
    p=histogram_probabilities(np.array([-100,1,100]),ref['edges']['near_peak'])
    assert p.sum()==pytest.approx(1)
    assert p[0]>p[1] and p[-1]>p[1]
    compare_distributions(np.ones((3,5))*100,np.zeros((7,5)),ref)
    assert repr(ref)==old
    assert ref['constant']==[True]*5


def test_frechet_analytic_singular_and_invalid_covariances():
    from cgan_research.metrics import gaussian_frechet
    assert gaussian_frechet(np.array([0.]),np.array([[4.]]),np.array([3.]),np.array([[9.]]))==pytest.approx(10.)
    assert gaussian_frechet(np.zeros(2),np.diag([1.,0]),np.zeros(2),np.diag([4.,0]))==pytest.approx(1.)
    with pytest.raises(ValueError):
        gaussian_frechet(np.zeros(2),np.diag([-1.,1]),np.zeros(2),np.eye(2))
    with pytest.raises(ValueError):
        gaussian_frechet(np.array([np.nan]),np.eye(1),np.zeros(1),np.eye(1))


def test_decimal_constant_features_are_excluded_exactly():
    from cgan_research.metrics import fit_metric_reference
    ref=fit_metric_reference(np.full((10,5),0.1))
    assert ref['constant']==[True]*5
    assert ref['scale']==[1.]*5
