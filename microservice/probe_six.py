"""Fresh shadow confirmation; frozen readouts and fixed prefixes only."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from microservice.partial_readout import score,diagnose
from microservice.latent_probe import labels,digest
from microservice.probe_target import frames

def load_readouts(root):
    config=json.loads((root/'config.json').read_text())
    assert hashlib.sha256((root/'readouts.npz').read_bytes()).hexdigest()==config['readouts_sha256']
    archive=np.load(root/'readouts.npz',allow_pickle=False)
    models={n:dict(config=c['config'],**{k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith(n+'__')}) for n,c in config['configs'].items()}
    return config,models

def stats(pred,rows):
    truth=diagnose(labels(rows));families=np.array([r['fault'] for r in rows])
    return dict(n=len(rows),correct=int(sum(pred==truth)),families={str(f):dict(n=int(sum(families==f)),correct=int(sum((pred==truth)&(families==f)))) for f in range(8)})

def score_rows(xs,rows,models):
    from scripts.evaluate_ledger_probe import features
    x={f'{n}{t+1}':xs[f'{n}_{t}'] for n in ('latent','history','summary_history') for t in (5,7)}
    x['response8']=features(rows)['ledger']
    predictions={n:diagnose(score(m,x[n])) for n,m in models.items()};truth=diagnose(labels(rows))
    metrics={n:stats(p,rows) for n,p in predictions.items()};paired={}
    a=predictions['latent6']
    for n,b in predictions.items():
        if n=='latent6':continue
        wins=int(sum((a==truth)&(b!=truth)));losses=int(sum((a!=truth)&(b==truth)));total=wins+losses
        p=min(1.,2*sum(math.comb(total,k) for k in range(min(wins,losses)+1))/2**total) if total else 1.
        paired[n]=dict(latent6_only_correct=wins,control_only_correct=losses,p=p)
    l=metrics['latent6'];reliable=l['correct']>=29 and l['families']['0']['correct']==4 and all(v['correct']>=3 for v in l['families'].values())
    matched=l['correct']>=metrics['latent8']['correct']-1 and l['correct']>=max(metrics[n]['correct'] for n in ('history8','summary_history8','response8'))-1
    requests={str(t):float(np.mean([sum(f['requests'] for f in frames(r)[:t]) for r in rows])) for t in (6,8)}
    result=dict(methods=metrics,paired=paired,reliable=bool(reliable),within_one_of_full_controls=bool(matched),feasibility_gate=bool(reliable and matched),
        shadow_mean_requests=requests,shadow_window_saving=.25,shadow_request_saving=1-requests['6']/requests['8'],
        request_saving_including_warmup=1-(requests['6']+36)/(requests['8']+36),actual_early_stopping=False)
    return result,dict(target=truth.tolist(),methods={n:p.tolist() for n,p in predictions.items()})

def main():
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    from microservice.probe_prefix import extract_prefixes
    from microservice.probe_six_collect import collect
    frozen,models=load_readouts(Path('frozen'));root=Path('/results/probe-target-model-v1/models')
    assert digest(root/'final-91')==frozen['checkpoint_sha256']
    assert hashlib.sha256(Path('collection/train.jsonl').read_bytes()).hexdigest()==frozen['train_sha256']
    assert hashlib.sha256(Path('reports/PROBE_SIX_PROTOCOL.md').read_bytes()).hexdigest()==frozen['protocol_sha256']
    out=Path('evaluation');out.mkdir(exist_ok=False);model=Model(str(root),91,'native')
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2));print('PROBE_SIX_FROZEN_VERIFIED',flush=True)
    start=time.time();rows=collect(out/'test.jsonl');assert len(rows)==32 and all(r['seed']==9302 for r in rows)
    xs=extract_prefixes(model,rows);np.savez_compressed(out/'test-features.npz',**xs)
    result,predictions=score_rows(xs,rows,models)
    assert digest(root/'final-91')==frozen['checkpoint_sha256'];load_readouts(Path('frozen'))
    result.update(checkpoint_unchanged=True,readouts_unchanged=True,collection_started_utc=start,frozen_utc=frozen['utc'])
    for n,v in [('summary',result),('predictions',predictions)]: (out/f'{n}.json').write_text(json.dumps(v,indent=2)+'\n')
    print('PROBE_SIX_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
