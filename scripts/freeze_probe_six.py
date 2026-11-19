"""Freeze fixed-six and comparison readouts using training evidence only."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.partial_readout import fit
from microservice.latent_probe import labels
from scripts.develop_forecast_readout import choose
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-six-v1'

def main():
    source=R/'reports/evidence/probe-stopping-v1';out=O/'frozen';out.mkdir(exist_ok=False)
    rows=[json.loads(l) for l in (O/'train.jsonl').read_text().splitlines()]
    assert len(rows)==64 and {r['seed'] for r in rows}=={9101,9103}
    fs=np.load(source/'evaluation/prefix-features.npz',allow_pickle=False)
    xs={f'{name}{t+1}':fs[f'{name}_{t}'] for name in ('latent','history','summary_history') for t in (5,7)}
    from scripts.evaluate_ledger_probe import features
    xs['response8']=features(rows)['ledger']
    y=labels(rows);groups=np.array([r['repeat'] for r in rows]);models={};configs={}
    for name,x in xs.items():
        config,counts=choose(x,y,groups);models[name]=fit(x,y,config)
        configs[name]=dict(config=config,cv_correct=counts)
    np.savez_compressed(out/'readouts.npz',**{f'{n}__{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
    frozen=dict(utc=time.time(),seed=9302,n=32,configs=configs,
        checkpoint_sha256=json.loads((source/'evaluation/extraction.json').read_text())['checkpoint_sha256'],
        train_sha256=hashlib.sha256((O/'train.jsonl').read_bytes()).hexdigest(),
        training_features_sha256=hashlib.sha256((source/'evaluation/prefix-features.npz').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((R/'reports/PROBE_SIX_PROTOCOL.md').read_bytes()).hexdigest(),
        readouts_sha256=hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest())
    (out/'config.json').write_text(json.dumps(frozen,indent=2)+'\n');print('FROZEN',list(models))
if __name__=='__main__':main()
