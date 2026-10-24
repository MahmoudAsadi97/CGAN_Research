"""Frozen group-disjoint preparation and train-only global min/max scaling."""
from dataclasses import asdict,is_dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .types import PreparedData
from .download import sha256_file


def encode_conditions(conditions,mapping):
    labels=np.asarray([mapping.get(str(c),-1) for c in conditions],dtype=np.int64)
    return labels,labels>=0


def transform_loads(physical,normalizer):
    values=np.asarray(physical,dtype=np.float64)
    if not np.isfinite(values).all(): raise ValueError('Physical loads must be finite')
    scale=float(normalizer['scale'])
    if not np.isfinite(scale) or scale<=0: raise ValueError('Normalizer scale must be positive and finite')
    return ((values-float(normalizer['min']))/scale).astype(np.float32)


def inverse_loads(normalized,normalizer):
    return np.asarray(normalized,dtype=np.float64)*float(normalizer['scale'])+float(normalizer['min'])


def _digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _digest_arrays(arrays):
    digest=hashlib.sha256()
    for name in sorted(arrays):
        arr=np.ascontiguousarray(arrays[name])
        digest.update(name.encode()); digest.update(str(arr.dtype).encode()); digest.update(str(arr.shape).encode()); digest.update(arr.tobytes())
    return digest.hexdigest()


def prepare_dataset(data,config,output_dir):
    """70/15/15 group shuffle, seed 42, floor train/validation and remainder test.

    Each partition must have at least one grouping unit. Identical numerical
    profiles are legitimate and retained; duplicate IDs or meter/date records
    are rejected. The manifest contains internal identifiers and is private.
    """
    output_dir=Path(output_dir)
    if output_dir.exists(): raise FileExistsError('Prepared output is immutable: '+str(output_dir))
    if not len(data.loads) or not np.isfinite(data.loads).all(): raise ValueError('Preparation requires nonempty finite physical loads')
    if np.any(data.household_ids==''): raise ValueError('Missing household/group IDs blocks the default split; revise protocol explicitly')
    if np.any(data.dates=='') or np.any(data.conditions==''): raise ValueError('Missing dates/calendar conditions block calendar conditioning')
    if len(np.unique(data.profile_ids))!=len(data.loads): raise ValueError('duplicate profile IDs')
    keys=list(zip(data.household_ids.tolist(),data.dates.tolist()))
    if len(set(keys))!=len(keys): raise ValueError('duplicate household/group-date measurements')
    settings=asdict(config) if is_dataclass(config) else dict(config)
    settings=settings.get('preparation',settings)
    seed=int(settings.get('split_seed',42))
    fractions=settings.get('split_fractions',[.7,.15,.15])
    if len(fractions)!=3 or not np.isclose(sum(fractions),1) or min(fractions)<=0: raise ValueError('Three positive split fractions summing to one required')
    groups=np.unique(data.household_ids)
    if len(groups)<3: raise ValueError('At least three household/group IDs required for nonempty partitions')
    shuffled=groups.copy(); np.random.default_rng(seed).shuffle(shuffled)
    ntrain=max(1,int(np.floor(len(groups)*fractions[0])))
    nval=max(1,int(np.floor(len(groups)*fractions[1])))
    if ntrain+nval>=len(groups): ntrain=len(groups)-nval-1
    if ntrain<1: raise ValueError('Split fractions cannot form three nonempty group partitions')
    assignments={'train':shuffled[:ntrain],'validation':shuffled[ntrain:ntrain+nval],'test':shuffled[ntrain+nval:]}
    splits={k:np.flatnonzero(np.isin(data.household_ids,v)).astype(np.int64) for k,v in assignments.items()}
    train=data.loads[splits['train']]; minimum=float(train.min()); maximum=float(train.max())
    normalizer={'min':minimum,'scale':maximum-minimum if maximum>minimum else 1.,'constant':maximum==minimum}
    mapping={c:i for i,c in enumerate(sorted(set(data.conditions[splits['train']].tolist())))}
    labels,supported=encode_conditions(data.conditions,mapping)
    normalized=transform_loads(data.loads,normalizer)
    arrays={'profile_ids':data.profile_ids,'household_ids':data.household_ids,'dates':data.dates,'conditions':data.conditions,'loads':data.loads.astype(np.float64),'normalized':normalized,'labels':labels,**{'split_'+k:v for k,v in splits.items()}}
    data_fingerprint=_digest_arrays({k:arrays[k] for k in ('profile_ids','household_ids','dates','conditions','loads')})
    split_record={k:{'household_ids':v.tolist(),'indices':splits[k].tolist()} for k,v in assignments.items()}
    manifest={'format_version':1,'units':data.units,'representations':{'loads':'physical','normalized':'global_train_minmax'},'dtypes':{'loads':'float64','normalized':'float32','labels':'int64'},'source_fingerprint':data.source_fingerprint,'data_fingerprint':data_fingerprint,'split_fingerprint':_digest_json(split_record),'split_seed':seed,'split_fractions':list(fractions),'split_rounding':'floor train and validation; remaining groups test; minimum one group per partition','splits':split_record,'normalizer':normalizer,'training_max':maximum,'condition_mapping':mapping,'unsupported_conditions':{k:int((~supported[v]).sum()) for k,v in splits.items()},'out_of_training_range':{k:int(((data.loads[v]<minimum)|(data.loads[v]>maximum)).sum()) for k,v in splits.items()},'profile_ids':data.profile_ids.tolist(),'household_assignments':{k:v.tolist() for k,v in assignments.items()},'synthetic':bool(data.metadata.get('synthetic',False)),'provenance':data.metadata,'privacy':'Contains internal meter/group identifiers; keep this artifact outside Git and public reports','counts':{k:{'profiles':len(splits[k]),'groups':len(assignments[k])} for k in splits},'arrays_content_fingerprint':_digest_arrays(arrays)}
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'.',dir=output_dir.parent))
    try:
        np.savez_compressed(stage/'arrays.npz',**arrays)
        manifest['arrays_sha256']=sha256_file(stage/'arrays.npz')
        manifest['manifest_fingerprint']=_digest_json(manifest)
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        stage.rename(output_dir)
    except BaseException:
        shutil.rmtree(stage,ignore_errors=True); raise
    return PreparedData(data.profile_ids,data.household_ids,data.dates,data.loads,normalized,labels,splits,manifest,data.conditions)


def load_prepared(path):
    path=Path(path); path=path.parent if path.name=='manifest.json' else path
    manifest=json.loads((path/'manifest.json').read_text())
    claimed=manifest.get('manifest_fingerprint'); actual=_digest_json({k:v for k,v in manifest.items() if k!='manifest_fingerprint'})
    if claimed!=actual: raise ValueError('Manifest fingerprint mismatch')
    if sha256_file(path/'arrays.npz')!=manifest['arrays_sha256']: raise ValueError('Arrays hash mismatch')
    with np.load(path/'arrays.npz',allow_pickle=False) as store: arrays={k:store[k] for k in store.files}
    if _digest_arrays(arrays)!=manifest['arrays_content_fingerprint']: raise ValueError('Arrays content fingerprint mismatch')
    if arrays['loads'].dtype!=np.float64 or arrays['normalized'].dtype!=np.float32 or arrays['labels'].dtype!=np.int64: raise ValueError('Prepared representation dtypes mismatch')
    splits={k:arrays['split_'+k] for k in ('train','validation','test')}
    return PreparedData(arrays['profile_ids'],arrays['household_ids'],arrays['dates'],arrays['loads'],arrays['normalized'],arrays['labels'],splits,manifest,arrays['conditions'])
