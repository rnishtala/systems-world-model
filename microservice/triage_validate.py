"""Freeze residual-assisted diagnosis readouts before collecting fresh tests."""
import argparse,json,time,math
from pathlib import Path
import numpy as np
from microservice.triage_audit import PREFIXES,baseline_fit,control_predictions,features
from microservice.forecast import CUTS,Forecaster,examples
from microservice.probe_target import vector
from microservice.readout import choose,fit,score,diagnose,labels,stats,digest
from microservice.validate import read_rows,sha,check_test
METHODS=('telemetry','telemetry_simple_residuals','telemetry_dreamer_residuals')

def inputs(rows,prefix,name,simple,dreamer):
    if name=='telemetry':return np.array([[vector(row,t) for t in range(prefix)] for row in rows]).reshape(len(rows),-1)
    return features(rows,prefix,simple if name=='telemetry_simple_residuals' else dreamer)

def freeze(rows,simple,dreamer,out,metadata):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    groups=np.array([r['repeat']%4 for r in rows]);y=labels(rows)
    if set(groups)!={0,1,2,3}:raise ValueError('Need all four training groups')
    arrays={};configs={}
    for prefix in PREFIXES:
        for name in METHODS:
            key=f'{prefix}_{name}';x=inputs(rows,prefix,name,simple,dreamer)
            if not np.isfinite(x).all():raise ValueError('Nonfinite training features')
            config,counts=choose(x,y,groups);model=fit(x,y,config)
            configs[key]=dict(config=config,cv_correct=counts)
            arrays.update({key+'__'+k:v for k,v in model.items() if k!='config'})
    np.savez_compressed(out/'readouts.npz',**arrays)
    frozen=dict(metadata,configs=configs,readouts_sha256=sha(out/'readouts.npz'),training_seeds=sorted({r['seed'] for r in rows}),utc=time.time())
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    return frozen

def load(path):
    path=Path(path);f=json.loads((path/'frozen.json').read_text())
    if sha(path/'readouts.npz')!=f['readouts_sha256']:raise ValueError('Readouts changed')
    with np.load(path/'readouts.npz',allow_pickle=False) as a:
        models={n:dict(config=c['config'],**{k.split('__',1)[1]:a[k] for k in a.files if k.startswith(n+'__')}) for n,c in f['configs'].items()}
    return f,models

def paired(candidate,control,truth):
    a=np.asarray(candidate)==truth;b=np.asarray(control)==truth
    wins=int(sum(a&~b));losses=int(sum(~a&b));n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return dict(candidate_only_correct=wins,control_only_correct=losses,exact_p=p)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['fit','evaluate'])
    for n in ('data','models','out'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--readouts');a=p.parse_args()
    rows=read_rows(a.data);cp=Path(a.models)/f'final-{a.seed}';before=digest(cp);data_hash=sha(a.data)
    if a.phase=='fit':
        meta=json.loads((Path(a.models)/f'training-{a.seed}.json').read_text())
        if meta['calibration_sha256']!=data_hash:raise ValueError('Wrong training data')
        baseline,selected,oof=baseline_fit(rows)
        simple=np.array([oof[selected[str(c)]][i*4+j] for i in range(len(rows)) for j,c in enumerate(CUTS)]).reshape(-1,4)
    else:
        if not a.readouts:raise ValueError('Frozen readouts required')
        frozen,models=load(a.readouts);check_test(rows,frozen)
        if before!=frozen['checkpoint_sha256']:raise ValueError('Checkpoint mismatch')
        if sha(Path(a.readouts)/'baseline.npz')!=frozen['baseline_sha256']:raise ValueError('Baseline changed')
        with np.load(Path(a.readouts)/'baseline.npz',allow_pickle=False) as f:baseline=dict(f)
        selected=frozen['selected_baseline_by_prefix'];simple,target=control_predictions(rows,baseline,selected)
    from microservice.model import Model
    forecaster=Forecaster(Model(a.models,a.seed));hs,_,_,_=examples(rows);pred=[]
    for i,h in enumerate(hs):
        pred.append(forecaster(h))
        if i%128==0:print('TRIAGE_VALIDATE_PROGRESS',a.phase,i,'/',len(hs),flush=True)
    dreamer=np.array(pred).reshape(-1,4)
    if not np.isfinite(dreamer).all():raise ValueError('Nonfinite model forecasts')
    out=Path(a.out)
    if a.phase=='fit':
        frozen=freeze(rows,simple,dreamer,out,dict(checkpoint_sha256=before,train_sha256=data_hash,selected_baseline_by_prefix=selected,primary_prefix=6))
        np.savez(out/'baseline.npz',**baseline)
        frozen.update(baseline_sha256=sha(out/'baseline.npz'),utc=time.time())
        (out/'frozen.json').write_text(json.dumps(frozen,indent=2))
    else:
        result=dict(kind='triage',prospective=True,scope='Fresh incidents collected after readouts froze; fixed six-observation primary comparison',primary_prefix=6,actual_early_stopping=False,prefixes={},paired={})
        saved={};truth=diagnose(labels(rows))
        for prefix in PREFIXES:
            result['prefixes'][str(prefix)]={};saved[str(prefix)]={}
            for name in METHODS:
                x=inputs(rows,prefix,name,simple,dreamer)
                if not np.isfinite(x).all():raise ValueError('Nonfinite test features')
                pred=diagnose(score(models[f'{prefix}_{name}'],x));saved[str(prefix)][name]=pred.tolist()
                metric=stats(pred,rows);metric['false_pool']=int(sum(((pred&2)>0)&((truth&2)==0)))
                result['prefixes'][str(prefix)][name]=metric
        primary=saved['6'];result['paired']={m:paired(primary[METHODS[2]],primary[m],truth) for m in METHODS[:2]}
        result['forecast']={k:float(abs(v-target).mean()) for k,v in [('dreamer',dreamer),('training_selected_baseline',simple)]}
        result['provenance']=dict(test_sha256=data_hash,checkpoint_sha256=before,readouts_sha256=frozen['readouts_sha256'])
        load(a.readouts)
        if sha(Path(a.readouts)/'baseline.npz')!=frozen['baseline_sha256']:raise ValueError('Baseline changed')
        out.mkdir(parents=True,exist_ok=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2))
        (out/'predictions.json').write_text(json.dumps(dict(target=truth.tolist(),methods=saved,test_sha256=data_hash)))
    if digest(cp)!=before or sha(a.data)!=data_hash:raise ValueError('Source artifacts changed')
    print('TRIAGE_VALIDATE_COMPLETE',a.phase,flush=True)

if __name__=='__main__':main()
