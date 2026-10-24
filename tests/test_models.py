import pytest
import torch


def test_reference_network_shapes_and_label_validation():
    from cgan_research.models import Generator, Discriminator
    g, d = Generator(3), Discriminator(3)
    x, y = torch.randn(7, 24), torch.tensor([0, 1, 2, 0, 1, 2, 0])
    assert g(x, y).shape == (7, 24)
    assert d(x, y).shape == (7, 1)
    assert [(m.in_features, m.out_features) for m in g.modules() if isinstance(m, torch.nn.Linear)] == [(27,64),(64,128),(128,64),(64,24)]
    assert [(m.in_features, m.out_features) for m in d.modules() if isinstance(m, torch.nn.Linear)] == [(27,64),(64,128),(128,256),(256,1)]
    for bad in (torch.full((7,), -1), torch.full((7,), 3), y.float()):
        with pytest.raises(ValueError):
            g(x, bad)
    with pytest.raises(ValueError):
        d(torch.ones(7, 23), y)
