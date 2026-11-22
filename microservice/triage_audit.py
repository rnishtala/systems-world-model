"""Training-selected forecast controls and retrospective residual-assisted triage."""
import argparse,json,time
from pathlib import Path
import numpy as np
from microservice.forecast import CUTS,Forecaster,examples,fit,predict,controls,PENALTIES
from microservice.probe_target import vector
from microservice.readout import choose,fit as fit_readout,score,diagnose,labels,stats,digest
from microservice.validate import read_rows,sha
PREFIXES=(4,5,6,7,8)

def baseline_fit(rows):
    hs,x,y,groups=examples(rows);oof=[]
    for penalty in PENALTIES:
        pred=np.zeros(len(y))
        for group in sorted(set(groups)):
            held=groups==group;pred[held]=predict(fit(x[~held],y[~held],penalty),x[held])
        oof.append(pred)
    best=int(np.argmin([abs(v-y).mean() for v in oof]))
    candidates={k:np.array([controls(h)[k] for h in hs]) for k in controls(hs[0])}
    candidates['regression']=oof[best]
    selected={str(c):min(candidates,key=lambda k:abs(candidates[k][i::4]-y[i::4]).mean()) for i,c in enumerate(CUTS)}
    return fit(x,y,PENALTIES[best]),selected,candidates

def control_predictions(rows,model,selected):
    hs,x,y,_=examples(rows)
    allpred={k:np.array([controls(h)[k] for h in hs]) for k in controls(hs[0])}
    allpred['regression']=predict(model,x)
    chosen=np.array([allpred[selected[str(c)]][i*4+j] for i in range(len(rows)) for j,c in enumerate(CUTS)])
    return chosen.reshape(-1,4),y.reshape(-1,4)

def features(rows,prefix,forecasts):
    # At prefix 6, only forecasts made after windows 4 and 5 have outcomes.
    if prefix not in PREFIXES:raise ValueError('Unsupported triage prefix')
    history=np.array([[vector(row,t) for t in range(prefix)] for row in rows]).reshape(len(rows),-1)
    count=max(0,prefix-4)
    actual=np.array([[vector(row,c)[0] for c in CUTS[:count]] for row in rows])
    used=np.asarray(forecasts)[:,:count]
    return np.concatenate([history,used,actual-used],axis=1)

def classify(train,test,train_x,test_x):
    groups=np.array([r['repeat']%4 for r in train]);y=labels(train)
    config,cv=choose(train_x,y,groups)
    pred=diagnose(score(fit_readout(train_x,y,config),test_x));truth=diagnose(labels(test))
    metrics=stats(pred,test);metrics['false_pool']=int(sum(((pred&2)>0)&((truth&2)==0)))
    return metrics,pred,dict(config=config,cv_correct=cv)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('models','train','test','out','original-predictions'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    cp=Path(a.models)/f'final-{a.seed}';before=digest(cp)
    train=read_rows(a.train);test=read_rows(a.test)
    if {r['seed'] for r in train}&{r['seed'] for r in test}:raise ValueError('Overlapping seeds')
    meta=json.loads((Path(a.models)/f'training-{a.seed}.json').read_text())
    if meta['calibration_sha256']!=sha(a.train):raise ValueError('Wrong training data')
    original=json.loads(Path(a.original_predictions).read_text())
    if original['test_sha256']!=sha(a.test):raise ValueError('Wrong retained test')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    protocol=dict(scope='Retrospective selection on source training only; retained test is exploratory',utc=time.time(),primary_prefix=6,training_sha256=sha(a.train),test_sha256=sha(a.test),checkpoint_sha256=before,planned_prefixes=list(PREFIXES),actual_early_stopping=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    baseline,selected,oof=baseline_fit(train)
    simple_train,_=control_predictions(train,baseline,selected);simple_test,target=control_predictions(test,baseline,selected)
    # Use OOF simple predictions for fitting the diagnostic readout.
    simple_train=np.array([oof[selected[str(c)]][i*4+j] for i in range(len(train)) for j,c in enumerate(CUTS)]).reshape(-1,4)
    from microservice.model import Model
    model=Model(a.models,a.seed);forecaster=Forecaster(model)
    predictions=[]
    for split,rows in [('train',train),('retained_test',test)]:
        hs,_,_,_=examples(rows);pred=[]
        for i,h in enumerate(hs):
            pred.append(forecaster(h))
            if i%128==0:print('TRIAGE_PROGRESS',split,i,'/',len(hs),flush=True)
        predictions.append(np.array(pred).reshape(-1,4))
    train_pred,test_pred=predictions
    if np.max(abs(test_pred.ravel()-original['methods']['dreamer']))>1e-5:raise ValueError('Frozen corrected forecasts changed')
    result=dict(**protocol,kind='triage',selected_baseline_by_prefix=selected,forecast={k:dict(log_mae=float(abs(v-target).mean()),by_prefix={str(c):float(abs(v[:,i]-target[:,i]).mean()) for i,c in enumerate(CUTS)}) for k,v in [('dreamer',test_pred),('training_selected_baseline',simple_test)]},prefixes={})
    configs={};saved={}
    for prefix in PREFIXES:
        result['prefixes'][str(prefix)]={};configs[str(prefix)]={};saved[str(prefix)]={}
        for name,tr,te in [('telemetry',None,None),('telemetry_simple_residuals',simple_train,simple_test),('telemetry_dreamer_residuals',train_pred,test_pred)]:
            if tr is None:
                x=np.array([[vector(row,t) for t in range(prefix)] for row in train]).reshape(len(train),-1)
                z=np.array([[vector(row,t) for t in range(prefix)] for row in test]).reshape(len(test),-1)
            else:x=features(train,prefix,tr);z=features(test,prefix,te)
            metrics,pred,config=classify(train,test,x,z)
            result['prefixes'][str(prefix)][name]=metrics;configs[str(prefix)][name]=config;saved[str(prefix)][name]=pred.tolist()
        print('TRIAGE_PREFIX_COMPLETE',prefix,flush=True)
    if digest(cp)!=before or sha(a.train)!=protocol['training_sha256'] or sha(a.test)!=protocol['test_sha256']:raise ValueError('Source artifacts changed')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'readout_selection.json').write_text(json.dumps(configs,indent=2))
    (out/'predictions.json').write_text(json.dumps(dict(target=diagnose(labels(test)).tolist(),methods=saved)))
    print('TRIAGE_COMPLETE',flush=True)

if __name__=='__main__':main()
