"""Explicit, bounded command interface; all data/output paths are supplied."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from .checkpoint import write_json


def read_config(path):
    value=yaml.safe_load(Path(path).read_text())
    if not isinstance(value,dict):
        raise ValueError('Configuration must be a YAML mapping')
    return value


def save_generated(batch,path):
    path=Path(path)
    if path.suffix!='.npz': raise ValueError('Generated output must have .npz suffix')
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    metadata={k:v for k,v in asdict(batch).items() if k not in ('loads','labels')}
    with path.open('xb') as stream:
        np.savez_compressed(stream,loads=batch.loads,labels=batch.labels,metadata=json.dumps(metadata))


def load_generated(path):
    from .training import GeneratedBatch
    with np.load(path,allow_pickle=False) as store:
        metadata=json.loads(str(store['metadata']))
        return GeneratedBatch(loads=store['loads'],labels=store['labels'],**metadata)


def parser():
    p=argparse.ArgumentParser(description='Electricity-only calendar-conditioned cGAN research')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('download'); s.add_argument('--url',required=True); s.add_argument('--output',required=True); s.add_argument('--sha256',required=True)
    for name in ('audit','prepare'):
        s=sub.add_parser(name); s.add_argument('--input',required=True); s.add_argument('--config',required=True); s.add_argument('--output',required=True)
        s.add_argument('--kind',choices=['daily','electricity'],required=True)
    s=sub.add_parser('train'); s.add_argument('--prepared',required=True); s.add_argument('--config',required=True); s.add_argument('--output',required=True); s.add_argument('--resume')
    s=sub.add_parser('generate'); group=s.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint'); group.add_argument('--empirical',action='store_true')
    s.add_argument('--prepared'); s.add_argument('--experiment-seed',type=int); s.add_argument('--samples-per-condition',type=int,default=2000); s.add_argument('--seed',type=int,default=42); s.add_argument('--output',required=True)
    s=sub.add_parser('evaluate'); s.add_argument('--prepared',required=True); s.add_argument('--generated',required=True); s.add_argument('--output',required=True)
    s.add_argument('--split',choices=['validation','test'],default='validation'); s.add_argument('--confirm-test',action='store_true',help='Record deliberate use of the frozen test partition')
    s.add_argument('--reference-replicates',type=int,default=100)
    s=sub.add_parser('plot'); s.add_argument('--evaluation',required=True); s.add_argument('--output',required=True); s.add_argument('--config')
    s=sub.add_parser('compare'); s.add_argument('--evaluations',nargs='+',required=True); s.add_argument('--output',required=True)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='download':
            from .download import download_asset
            result=download_asset(args.url,args.output,args.sha256)
            print(json.dumps(asdict(result))); return 0
        if args.command in ('audit','prepare'):
            from .data import read_daily_csv,read_electricity_archive
            from .prepare import prepare_dataset
            from .metrics import fit_reference_from_prepared
            output=Path(args.output)
            if output.exists(): raise FileExistsError(output)
            cfg=read_config(args.config)
            reader=read_daily_csv if args.kind=='daily' else read_electricity_archive
            data,audit=reader(args.input,cfg)
            if not audit.accepted: raise ValueError('No valid profiles remain after audit')
            if args.command=='audit':
                output.parent.mkdir(parents=True,exist_ok=True)
                write_json(output,audit.to_dict())
            else:
                prepared=prepare_dataset(data,cfg,output)
                write_json(output/'audit.json',audit.to_dict())
                fit_reference_from_prepared(prepared,output_path=output/'metric_reference.json')
            print(json.dumps({'accepted':audit.accepted,'quarantined':audit.quarantined,'output':str(output)})); return 0
        if args.command=='train':
            from .prepare import load_prepared
            from .training import train,TrainConfig
            cfg=read_config(args.config)
            result=train(load_prepared(args.prepared),TrainConfig(**cfg.get('training',cfg)),args.output,resume=args.resume)
            print(json.dumps(result)); return 0
        if args.command=='generate':
            from .training import generate
            from .checkpoint import load_checkpoint
            if args.empirical:
                if not args.prepared: raise ValueError('--empirical requires --prepared')
                from .evaluation import empirical_baseline
                from .prepare import load_prepared
                batch=empirical_baseline(load_prepared(args.prepared),args.samples_per_condition,args.seed,args.experiment_seed)
            else:
                if args.experiment_seed is not None: raise ValueError('--experiment-seed is only for the empirical baseline; cGAN training seed comes from its checkpoint')
                checkpoint=load_checkpoint(args.checkpoint)
                batch=generate(args.checkpoint,list(range(len(checkpoint['condition_mapping']))),args.samples_per_condition,args.seed)
            save_generated(batch,args.output)
            print(json.dumps({'generated':len(batch.loads),'units':batch.units,'representation':batch.representation})); return 0
        if args.command=='evaluate':
            from .prepare import load_prepared
            from .metrics import fit_reference_from_prepared
            from .evaluation import evaluate,save_evaluation
            if args.split=='test' and not args.confirm_test: raise ValueError('Frozen test evaluation requires explicit --confirm-test; pilots use validation')
            prepared=load_prepared(args.prepared)
            ref_path=Path(args.prepared)/'metric_reference.json'
            reference=json.loads(ref_path.read_text()) if ref_path.exists() else fit_reference_from_prepared(prepared)
            result=evaluate(prepared,load_generated(args.generated),reference,args.split,args.reference_replicates)
            result['metadata']['test_opened']=args.split=='test'
            save_evaluation(result,args.output)
            print(json.dumps({'split':args.split,'metric_rows':len(result['rows']),'output':args.output})); return 0
        if args.command=='plot':
            from .figures import render_figures
            result=render_figures(args.evaluation,args.output,read_config(args.config) if args.config else {})
            print(json.dumps(result)); return 0
        if args.command=='compare':
            from .evaluation import compare_runs
            path=Path(args.output)
            if path.exists(): raise FileExistsError(path)
            path.parent.mkdir(parents=True,exist_ok=True)
            write_json(path,compare_runs(args.evaluations)); return 0
    except (ValueError,TypeError,KeyError,OSError,RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        return 2
    return 0
