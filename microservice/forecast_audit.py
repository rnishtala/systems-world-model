"""Retrospective frozen-model diagnostic; no fitting or new fault collection."""
import argparse,json
from pathlib import Path
import numpy as np
from microservice.forecast import CUTS,Forecaster,history
from microservice.probe_target import vector
from microservice.validate import read_rows,sha

def metrics(actual,pred):
    actual=np.asarray(actual);pred=np.asarray(pred)
    if actual.shape!=pred.shape or not np.isfinite(pred).all():raise ValueError('Invalid diagnostic predictions')
    error=pred-actual
    return dict(log_mae=float(abs(error).mean()),bias=float(error.mean()),milliseconds_mae=float(abs(100*np.expm1(np.maximum(0,pred))-100*np.expm1(actual)).mean()))

def summarize(rows,actual,pred):
    actual=np.asarray(actual);pred=np.asarray(pred)
    if actual.shape!=(len(rows),len(CUTS),2) or pred.shape!=actual.shape:raise ValueError('Invalid diagnostic dimensions')
    result={}
    for j,name in enumerate(('posterior_reconstruction','prior_forecast')):
        result[name]=metrics(actual[:,:,j],pred[:,:,j])
        result[name]['by_prefix']={str(c):metrics(actual[:,i,j],pred[:,i,j]) for i,c in enumerate(CUTS)}
        result[name]['by_fault']={str(f):metrics(actual[[r['fault']==f for r in rows],:,j],pred[[r['fault']==f for r in rows],:,j]) for f in sorted({r['fault'] for r in rows})}
    return dict(incidents=len(rows),methods=result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('models','train','test','out','original-predictions'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    from microservice.model import Model
    from microservice.readout import digest
    cp=Path(a.models)/f'final-{a.seed}';before=digest(cp)
    paths=[Path(a.train),Path(a.test),Path(a.original_predictions)];hashes={str(p):sha(p) for p in paths}
    model=Model(a.models,a.seed);predict=Forecaster(model,diagnostic=True)
    result=dict(scope='Retrospective diagnostic; reconstruction sees its target and is not a forecast baseline',checkpoint_sha256=before,data_sha256=hashes,cuts=list(CUTS),particles=16,splits={})
    for split,path in [('train',a.train),('retained_test',a.test)]:
        rows=read_rows(path);actual=[];pred=[]
        for i,row in enumerate(rows):
            actual.append([[float(vector(row,c-1)[0]),float(vector(row,c)[0])] for c in CUTS])
            pred.append([predict(history(row,c)).tolist() for c in CUTS])
            if i%16==0:print('AUDIT_PROGRESS',split,i,'/',len(rows),flush=True)
        result['splits'][split]=summarize(rows,actual,np.asarray(pred)[:,:,:2])
        if split=='retained_test':
            original=json.loads(Path(a.original_predictions).read_text())
            if original['test_sha256']!=sha(path):raise ValueError('Original test identity mismatch')
            delta=float(np.max(abs(np.asarray(pred)[:,:,2].ravel()-original['methods']['dreamer'])))
            result['prior_reproduction_max_abs_error']=delta
            if delta>1e-5:raise ValueError('Diagnostic changed raw prior forecasts')
            from microservice.forecast import summarize as score_forecast
            methods=dict(original['methods'],dreamer=np.asarray(pred)[:,:,1].ravel().tolist())
            result['corrected_forecast']=score_forecast(original['target'],methods,'regression',len(rows))
            result['correction']='Invert decoder symlog per particle before averaging; original raw outputs reproduce exactly'
            result['test_status']='Retrospective corrected scoring, requires fresh confirmation'
    if digest(cp)!=before or any(sha(p)!=hashes[str(p)] for p in paths):raise ValueError('Source artifact changed')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('AUDIT_COMPLETE',flush=True)

if __name__=='__main__':main()
