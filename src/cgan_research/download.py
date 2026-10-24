"""Stream verified assets and safely inspect/extract selected ZIP members."""
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from urllib.parse import urlparse
from urllib.request import Request,urlopen
import zipfile
from .types import AssetManifest


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def download_asset(url,destination,expected_sha256=None):
    """HTTPS only; verify bytes before an atomic, non-overwriting final rename."""
    parsed=urlparse(url)
    if parsed.scheme!='https': raise ValueError('Asset download requires HTTPS')
    if (parsed.hostname=='github.com' and ('/archive/' in parsed.path or '/zipball/' in parsed.path)) or parsed.hostname=='codeload.github.com':
        raise ValueError('Automatic source archives are not dataset assets; select a release asset URL')
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+destination.name+'.',suffix='.part',dir=destination.parent)
    digest=hashlib.sha256(); size=0
    try:
        with os.fdopen(fd,'wb') as out, urlopen(Request(url,headers={'User-Agent':'cgan-research/0.1'}),timeout=60) as response:
            if urlparse(response.geturl()).scheme!='https': raise ValueError('Redirect must preserve HTTPS')
            if 'html' in response.headers.get('Content-Type','').lower(): raise ValueError('HTML error response is not a dataset asset')
            first=True
            while True:
                chunk=response.read(1024*1024)
                if not chunk: break
                if first and chunk.lstrip().lower().startswith((b'<!doctype html',b'<html')): raise ValueError('HTML error response is not a dataset asset')
                first=False; digest.update(chunk); size+=len(chunk); out.write(chunk)
            out.flush(); os.fsync(out.fileno())
        if not size: raise ValueError('Empty asset response')
        actual=digest.hexdigest()
        if expected_sha256 and actual.lower()!=expected_sha256.lower(): raise ValueError('SHA-256 mismatch')
        # link() fails if another writer created the destination while downloading.
        os.link(tmp,destination)
        return AssetManifest(url,str(destination),size,actual,expected_sha256)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def validate_zip_members(archive,max_expanded_bytes=20_000_000_000):
    """Validate all names, collisions, link modes and advertised expansion bounds."""
    infos=archive.infolist(); seen=set(); total=0
    for info in infos:
        name=info.filename
        path=PurePosixPath(name)
        if not name or '\\' in name or path.is_absolute() or '..' in path.parts or re.match(r'^[A-Za-z]:',name):
            raise ValueError('unsafe ZIP member path')
        key=str(path).rstrip('/')
        if key in seen: raise ValueError('duplicate ZIP member path')
        seen.add(key)
        mode=info.external_attr>>16
        if stat.S_ISLNK(mode): raise ValueError('unsafe ZIP symbolic link')
        total+=info.file_size
        if total>max_expanded_bytes: raise ValueError('ZIP expanded-size ceiling exceeded')
    return infos


def safe_extract_zip(archive,destination,members=None,max_expanded_bytes=20_000_000_000):
    """Extract selected regular members after validating the entire archive."""
    destination=Path(destination)
    if destination.exists(): raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as z:
        infos=validate_zip_members(z,max_expanded_bytes)
        selected=set(members) if members is not None else {i.filename for i in infos}
        if selected-{i.filename for i in infos}: raise ValueError('Selected ZIP members do not exist')
        destination.parent.mkdir(parents=True,exist_ok=True)
        stage=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.',dir=destination.parent))
        try:
            written=0
            for info in infos:
                if info.filename not in selected: continue
                target=stage/str(PurePosixPath(info.filename))
                if info.is_dir(): target.mkdir(parents=True,exist_ok=True); continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as src,target.open('xb') as out:
                    while True:
                        chunk=src.read(1024*1024)
                        if not chunk: break
                        written+=len(chunk)
                        if written>max_expanded_bytes: raise ValueError('ZIP expanded-size ceiling exceeded')
                        out.write(chunk)
            stage.rename(destination)
        except BaseException:
            shutil.rmtree(stage,ignore_errors=True); raise
    return destination
