import pytest
import torch


def test_atomic_write_keeps_last_valid_checkpoint(tmp_path,monkeypatch):
    from cgan_research.checkpoint import save_checkpoint,load_checkpoint
    path=tmp_path/'last.pt'
    save_checkpoint(path,{'version':1,'data_fingerprint':'a','x':torch.ones(2)})
    def fail(*args,**kwargs):
        raise OSError('interrupted')
    monkeypatch.setattr(torch,'save',fail)
    with pytest.raises(OSError): save_checkpoint(path,{'x':torch.zeros(2)})
    assert load_checkpoint(path)['x'].tolist()==[1,1]
    with pytest.raises(ValueError,match='data_fingerprint'):
        load_checkpoint(path,{'data_fingerprint':'b'})
    path.write_bytes(b'partial')
    with pytest.raises(ValueError,match='checkpoint'):
        load_checkpoint(path)
