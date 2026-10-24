import numpy as np
import pytest


def test_quantiles_cyclic_times_and_flat_profiles():
    from cgan_research.features import daily_features
    x=np.stack([np.arange(24),np.ones(24),np.roll(np.arange(24),5)])
    f=daily_features(x)
    np.testing.assert_allclose(f[0],[22.425,.575,12,23,1])
    np.testing.assert_allclose(f[1],[1,1,0,0,0])
    np.testing.assert_allclose(f[2],f[0])
    compat=daily_features(x,mode='upstream_compatibility')
    np.testing.assert_allclose(compat[0],[22,1,12,22,22])
    x[0,1]=np.nan
    with pytest.raises(ValueError): daily_features(x)


def test_first_peak_tie_is_used():
    from cgan_research.features import daily_features
    x=np.zeros((1,24)); x[0,[1,2,5]]=10
    f=daily_features(x)[0]
    assert f[2:].tolist()==[3,1,2]
