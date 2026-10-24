import hashlib
import io
import zipfile
import pytest
from cgan_research.download import download_asset,safe_extract_zip

class Response(io.BytesIO):
    def __init__(self,payload,ctype='application/zip',url='https://example.org/raw.zip'):
        super().__init__(payload); self.headers={'Content-Type':ctype}; self.url=url
    def geturl(self): return self.url


def test_download_stream_atomic_checksum(tmp_path,monkeypatch):
    import cgan_research.download as m
    payload=b'PK\x03\x04fixture'
    monkeypatch.setattr(m,'urlopen',lambda request,**kw:Response(payload))
    p=tmp_path/'raw.zip'; result=download_asset('https://example.org/raw.zip',p,hashlib.sha256(payload).hexdigest())
    assert p.read_bytes()==payload and result.bytes==len(payload) and result.sha256==hashlib.sha256(payload).hexdigest()
    assert not list(tmp_path.glob('*.part'))
    q=tmp_path/'bad.zip'
    with pytest.raises(ValueError,match='SHA'): download_asset('https://example.org/raw.zip',q,'0'*64)
    assert not q.exists()


def test_download_rejects_html_autoarchive_and_insecure(tmp_path,monkeypatch):
    import cgan_research.download as m
    monkeypatch.setattr(m,'urlopen',lambda request,**kw:Response(b'<html>error</html>','text/html'))
    for url,match in [('https://example.org/raw.zip','HTML'),('https://github.com/x/y/archive/refs/heads/main.zip','source'),('http://example.org/raw.zip','HTTPS')]:
        with pytest.raises(ValueError,match=match): download_asset(url,tmp_path/'bad.zip')
    assert not (tmp_path/'bad.zip').exists()


def test_safe_zip_selected_and_rejects_unsafe(tmp_path):
    z=tmp_path/'good.zip'
    with zipfile.ZipFile(z,'w') as f: f.writestr('good/a.csv','ok'); f.writestr('ignore.txt','x')
    safe_extract_zip(z,tmp_path/'out',members=['good/a.csv'])
    assert (tmp_path/'out/good/a.csv').read_text()=='ok' and not (tmp_path/'out/ignore.txt').exists()
    with pytest.raises(ValueError,match='expanded'): safe_extract_zip(z,tmp_path/'small',max_expanded_bytes=1)
    for index,names in enumerate([['../escape'],['same','same'],['/absolute'],['C:/windows']]):
        p=tmp_path/f'bad{index}.zip'
        with pytest.warns(UserWarning) if index==1 else __import__('contextlib').nullcontext():
            with zipfile.ZipFile(p,'w') as f:
                for n in names: f.writestr(n,'x')
        with pytest.raises(ValueError,match='unsafe|duplicate'): safe_extract_zip(p,tmp_path/f'x{index}')
