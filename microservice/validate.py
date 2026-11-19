"""Fit frozen readouts, then evaluate on disjoint fresh incidents."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
from microservice.readout import choose,diagnose,digest,fit,labels,score,stats
from microservice.probe_target import frames

NAMES={f'{kind}{t+1}':f'{kind}_{t}' for kind in ('latent','history') for t in (5,7)}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_rows(path):
    rows=[json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows:raise ValueError('Empty dataset')
    for row in rows:frames(row)
    return rows

def check_test(rows,frozen):
    if set(r['seed'] for r in rows)&set(frozen['training_seeds']):
        raise ValueError('Training and test seeds overlap')
    if any(r['utc']<=frozen['utc'] for r in rows):
        raise ValueError('Test must be collected after readouts are frozen')
    if len(rows)!=64 or any(sum(r['fault']==f and r['regime']==g for r in rows)!=4 for f in range(8) for g in (0,1)):
        raise ValueError('Expected 64 balanced original/shifted test incidents')

def load_readouts(path):
    path=Path(path);frozen=json.loads((path/'frozen.json').read_text())
    if sha(path/'readouts.npz')!=frozen['readouts_sha256']:raise ValueError('Readouts changed')
    with np.load(path/'readouts.npz',allow_pickle=False) as arrays:
        models={n:dict(config=c['config'],**{k.split('__',1)[1]:arrays[k] for k in arrays.files if k.startswith(n+'__')}) for n,c in frozen['configs'].items()}
    return frozen,models

def summarize(pred,rows):
    truth=diagnose(labels(rows));methods={n:stats(p,rows) for n,p in pred.items()}
    regimes={str(g):{n:stats(p[np.array([r['regime']==g for r in rows])],[r for r in rows if r['regime']==g]) for n,p in pred.items()} for g in (0,1)}
    pairs={}
    for name,p in pred.items():
        if name=='latent6':continue
        a=pred['latent6']==truth;b=p==truth;wins=int(sum(a&~b));losses=int(sum(~a&b));total=wins+losses
        pairs[name]=dict(new_only_correct=wins,control_only_correct=losses,p=min(1.,2*sum(math.comb(total,k) for k in range(min(wins,losses)+1))/2**total) if total else 1.)
    false_pool={n:int(sum(((p&2)>0)&((truth&2)==0))) for n,p in pred.items()}
    bit_errors={n:np.sum(((p[:,None]>>np.arange(3))&1)!=(labels(rows)>0),axis=0).tolist() for n,p in pred.items()}
    primary=methods['latent6'];reliable=primary['correct']>=58 and primary['families']['0']['correct']==8 and all(v['correct']>=6 for v in primary['families'].values())
    regime_gate=all(m['latent6']['correct']>=29 and m['latent6']['families']['0']['correct']==4 and all(v['correct']>=3 for v in m['latent6']['families'].values()) for m in regimes.values())
    matched=primary['correct']>=max(methods[n]['correct'] for n in ('history6','latent8'))-2
    return dict(methods=methods,regimes=regimes,false_pool=false_pool,bit_errors=bit_errors,paired=pairs,reliable=bool(reliable),regime_gate=bool(regime_gate),matched_controls=bool(matched),actual_early_stopping=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['fit','evaluate']);p.add_argument('--data',required=True)
    p.add_argument('--models',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--readouts')
    p.add_argument('--baseline',help='Prior evaluation directory on byte-identical test data; compare its latent6')
    a=p.parse_args();rows=read_rows(a.data);root=Path(a.models);out=Path(a.out)
    checkpoint=root/f'final-{a.seed}'
    if not checkpoint.is_dir():p.error('Checkpoint missing')
    before=digest(checkpoint)
    if a.phase=='fit':
        if a.readouts or a.baseline:p.error('fit does not consume evaluation artifacts')
        meta=json.loads((root/f'training-{a.seed}.json').read_text())
        if meta['calibration_sha256']!=sha(a.data):p.error('Readout data must match designated model-training data')
        groups=np.array([r['repeat']%4 for r in rows])
        if len(set(groups))!=4:p.error('Need all four repeat groups for training-only selection')
    else:
        if not a.readouts:p.error('evaluate requires --readouts')
        frozen,models=load_readouts(a.readouts);check_test(rows,frozen)
        if frozen['checkpoint_sha256']!=before:p.error('Readouts belong to another checkpoint')
    out.mkdir(parents=True,exist_ok=False)
    from microservice.model import Model
    from microservice.auxiliary_audit import audit
    from microservice.probe_prefix import extract_prefixes
    model=Model(root,a.seed);(out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    xs=extract_prefixes(model,rows);xs={v:xs[v] for v in NAMES.values()}
    if not all(np.isfinite(x).all() for x in xs.values()):raise ValueError('Nonfinite features')
    np.savez_compressed(out/'features.npz',**xs)
    if a.phase=='fit':
        y=labels(rows);models={};configs={}
        for n,key in NAMES.items():
            config,counts=choose(xs[key],y,groups);models[n]=fit(xs[key],y,config);configs[n]=dict(config=config,cv_correct=counts)
        np.savez_compressed(out/'readouts.npz',**{f'{n}__{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
        frozen=dict(utc=time.time(),checkpoint_sha256=before,readouts_sha256=sha(out/'readouts.npz'),train_sha256=sha(a.data),training_seeds=sorted(set(r['seed'] for r in rows)),configs=configs)
        (out/'frozen.json').write_text(json.dumps(frozen,indent=2)+'\n')
    else:
        pred={n:diagnose(score(models[n],xs[key])) for n,key in NAMES.items()}
        if a.baseline:
            baseline=json.loads((Path(a.baseline)/'predictions.json').read_text())
            if baseline['test_sha256']!=sha(a.data) or baseline['target']!=diagnose(labels(rows)).tolist():raise ValueError('Baseline must use identical test incidents')
            pred['baseline6']=np.array(baseline['methods']['latent6'])
        summary=summarize(pred,rows)
        summary['improvement_gate']=bool(summary['methods']['latent6']['correct']>=summary['methods']['baseline6']['correct']+2 and summary['false_pool']['latent6']<summary['false_pool']['baseline6']) if a.baseline else None
        summary['feasibility_gate']=bool(summary['reliable'] and summary['regime_gate'] and summary['matched_controls'] and summary['improvement_gate']) if a.baseline else None
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'predictions.json').write_text(json.dumps(dict(test_sha256=sha(a.data),target=diagnose(labels(rows)).tolist(),methods={n:v.tolist() for n,v in pred.items()}),indent=2)+'\n')
        load_readouts(a.readouts)
    if digest(checkpoint)!=before:raise RuntimeError('Checkpoint changed during validation')
    (out/'provenance.json').write_text(json.dumps(dict(checkpoint_sha256=before,input_sha256=sha(a.data),checkpoint_unchanged=True),indent=2)+'\n')
    print(a.phase.upper()+'_COMPLETE',out)
if __name__=='__main__':main()
