"""Fresh incidents scored by immutable training-only readouts and checkpoint."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.agent import Model
from microservice.auxiliary_audit import audit
from microservice.workload import extract
from microservice.latent_probe import labels,digest
from microservice.partial_readout import score,diagnose
from microservice.workload_evaluate import stats
from microservice.forecast_readout import features
from microservice import regime_collect

def main():
    out=Path('evaluation');out.mkdir(exist_ok=False)
    frozen=json.loads(Path('frozen/frozen.json').read_text())
    h=hashlib.sha256(Path('frozen/readouts.npz').read_bytes()).hexdigest()
    assert h==frozen['readouts_sha256']
    root=Path('/results/lab-workload-v2/models');before=digest(root/'final-81')
    assert before==frozen['checkpoint_sha256']
    model=Model(str(root),81,'native')
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2)+'\n')
    (out/'before-collection.json').write_text(json.dumps(dict(utc=time.time(),checkpoint_sha256=before,readouts_sha256=h),indent=2)+'\n')
    regime_collect.SEEDS={**regime_collect.SEEDS,'validation':9002}
    rows=regime_collect.collect(out/'confirmation.jsonl','validation')
    base,aux,forecast=extract(model,rows)
    xs=features(base,dict(fixed=forecast['predicted'],current=forecast['persistence'],actual=forecast['actual']))
    np.savez_compressed(out/'features.npz',**xs,labels=labels(rows))
    np.savez_compressed(out/'forecast.npz',**forecast)
    arrays=np.load('frozen/readouts.npz');predictions={};metrics={}
    for name,selected in frozen['selection'].items():
        m={k:arrays[f'{name}_{k}'] for k in ('mean','scale','bias','z','dual')}
        m['config']=selected['config'];predictions[name]=diagnose(score(m,xs[name]))
        metrics[name]=stats(predictions[name],rows)
    predictions['auxiliary']=aux;metrics['auxiliary']=stats(aux,rows)
    target=diagnose(labels(rows))
    (out/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),methods={k:v.tolist() for k,v in predictions.items()}),indent=2)+'\n')
    mse={k:float(np.mean((forecast[k]-forecast['actual'])**2)) for k in ('predicted','persistence')}
    primary=metrics['latent']
    reliable=bool(primary['correct']>=58 and primary['families']['0']['correct']==8 and all(v['correct']>=6 for k,v in primary['families'].items() if k!='0'))
    advantage=bool(primary['correct']>=max(metrics[k]['correct'] for k in ('response','history','response_history'))+4)
    forecast_gate=mse['predicted']<mse['persistence']
    import math
    comparisons={}
    for name in ('response','history','response_history'):
        a=predictions['latent']==target;b=predictions[name]==target
        wins=int(sum(a&~b));losses=int(sum(~a&b));n=wins+losses
        p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        comparisons[name]=dict(latent_only_correct=wins,baseline_only_correct=losses,two_sided_exact_p=p)
    assert before==digest(root/'final-81')
    assert h==hashlib.sha256(Path('frozen/readouts.npz').read_bytes()).hexdigest()
    summary=dict(methods=metrics,forecast_mse=mse,primary_reliable=reliable,added_value=advantage,
        forecast_gate=forecast_gate,combined_gate=bool(reliable and advantage and forecast_gate),
        paired=comparisons,checkpoint_unchanged=True,readouts_unchanged=True,refit_after_collection=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('FROZEN_CONFIRMATION_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
