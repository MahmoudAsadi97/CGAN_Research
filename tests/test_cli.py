import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]


def command(tmp_path,*args):
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MPLCONFIGDIR':'/tmp/cgan-mpl','OPENBLAS_NUM_THREADS':'1'}
    return subprocess.run([sys.executable,'-m','cgan_research',*map(str,args)],cwd=tmp_path,env=env,capture_output=True,text=True)


def test_cli_synthetic_end_to_end_and_errors(tmp_path):
    dates=['2024-01-01','2024-01-06','2024-04-01','2024-04-06','2024-07-01','2024-07-06','2024-10-01','2024-10-05']
    rows=[]
    for h in range(20):
        for j,date in enumerate(dates):
            rows.append({'profile_id':f'{h}:{date}','household_id':str(h),'date':date,
                         **{f'load_{hour:02}':-2+0.1*h+0.2*j+np.sin(hour/3) for hour in range(24)}})
    data=tmp_path/'synthetic loads.csv'; pd.DataFrame(rows).to_csv(data,index=False)
    config=tmp_path/'dataset.yaml'; config.write_text(yaml.safe_dump({'dataset':{'synthetic':True}}))
    prepared=tmp_path/'prepared data'
    p=command(tmp_path,'prepare','--input',data,'--config',config,'--output',prepared,'--kind','daily')
    assert p.returncode==0,p.stderr
    assert json.loads((prepared/'manifest.json').read_text())['synthetic']
    cfg=tmp_path/'train.yaml'
    cfg.write_text(yaml.safe_dump({'training':{'device':'cpu','max_steps':8,'max_epochs':2,'batch_size':32,'validation_interval':2,'checkpoint_interval':2,'samples_per_condition':16}}))
    run=tmp_path/'run'
    p=command(tmp_path,'train','--prepared',prepared,'--config',cfg,'--output',run)
    assert p.returncode==0,p.stderr
    assert json.loads((run/'status.json').read_text())['completed_epochs']==2
    generated=tmp_path/'samples.npz'
    p=command(tmp_path,'generate','--checkpoint',run/'best.pt','--output',generated,'--samples-per-condition','16')
    assert p.returncode==0,p.stderr
    evaluation=tmp_path/'evaluation'
    p=command(tmp_path,'evaluate','--prepared',prepared,'--generated',generated,'--output',evaluation,'--reference-replicates','2')
    assert p.returncode==0,p.stderr
    r=json.loads((evaluation/'evaluation.json').read_text())
    assert len(r['rows'])==8*5*8 and r['metadata']['provenance']['synthetic']
    assert len(r['sensitivity'])==8*2*3
    for args in [('prepare','--input',data,'--config',config,'--output',prepared,'--kind','daily'),
                 ('train','--prepared',prepared,'--config',cfg,'--output',run),
                 ('evaluate','--prepared',prepared,'--generated',generated,'--output',tmp_path/'test','--split','test')]:
        assert command(tmp_path,*args).returncode!=0
    p=command(tmp_path,'plot','--evaluation',evaluation,'--output',tmp_path/'figures')
    assert p.returncode==0,p.stderr
    assert (tmp_path/'figures'/'figure_manifest.json').is_file()
