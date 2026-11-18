"""Nested development selection of inspection order, readout and stopping rule."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from microservice.order_features import ORDERS
from microservice.partial_readout import outer_fit,folds,assess,diagnose
from microservice.efficiency import decisions
from microservice.mechanisms import FAULT_MASK
KEYS=[''.join(map(str,o)) for o in ORDERS]

def choose_order(runs,target,families):
    candidates=[]
    for key in KEYS:
        run=runs[key];p=run['policy']
        stats=assess(*decisions(run['inner_scores'],p['threshold']),target,families)
        full=assess(*decisions(run['inner_scores'],None),target,families)
        candidates.append(dict(order=key,calibration_passed=p['calibration_passed'],threshold=p['threshold'],stopping=stats,full=full))
    eligible=[c for c in candidates if c['calibration_passed']]
    if eligible:
        best=min(eligible,key=lambda c:(c['stopping']['mean_checks'],-c['stopping']['correct'],KEYS.index(c['order'])))
        return best['order'],best['threshold'],candidates
    best=min(candidates,key=lambda c:(-c['full']['correct'],KEYS.index(c['order'])))
    return best['order'],None,candidates

def main():
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];source=root/'reports/evidence/lab-efficiency-v1/evaluation/development.jsonl';feature_path=Path(a.features)
    hashes=dict(development_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),features_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest())
    rows=[json.loads(s) for s in source.read_text().splitlines()];assert len(rows)==80 and all(r['seed']==8201 for r in rows)
    families=np.array([r['fault'] for r in rows]);target=np.array([FAULT_MASK[f] for f in families]);y=((target[:,None]>>np.arange(3))&1)*2-1
    fs=np.load(feature_path);outer=folds(families,5,8301);splits=[]
    for k,test in enumerate(outer):
        train=np.setdiff1d(np.arange(80),test);inner=folds(families[train],4,8302+k)
        splits.append(dict(train=train.tolist(),test=test.tolist(),inner=[dict(train=train[np.setdiff1d(np.arange(64),held)].tolist(),test=train[held].tolist()) for held in inner]))
    result={};predictions={}
    for name in ['latent','raw']:
        per_order={key:dict(scores=np.empty((80,4,3)),pred=np.empty(80,int),checks=np.empty(80,int)) for key in KEYS}
        selected_scores=np.empty((80,4,3));selected_pred=np.empty(80,int);selected_checks=np.empty(80,int);selected_orders=['']*80;details=[]
        for k,test in enumerate(outer):
            train=np.array(splits[k]['train']);inner=folds(families[train],4,8302+k);runs={}
            for key in KEYS:
                features=[fs[f'{name}_{key}_{t}'] for t in range(4)]
                run=outer_fit(features,y,families,train,test,inner);runs[key]=run
                per_order[key]['scores'][test]=run['scores'];per_order[key]['pred'][test],per_order[key]['checks'][test]=decisions(run['scores'],run['policy']['threshold'])
            chosen,threshold,candidates=choose_order(runs,target[train],families[train]);run=runs[chosen]
            selected_scores[test]=run['scores'];selected_pred[test],selected_checks[test]=decisions(run['scores'],threshold)
            for i in test:selected_orders[i]=chosen
            details.append(dict(chosen=chosen,threshold=threshold,candidates=candidates,runs={key:{field:v[field] for field in ('selected','policy','selection')} for key,v in runs.items()}))
            print('ORDER_OUTER_COMPLETE',name,k,chosen,threshold,flush=True)
        stats={}
        for key,d in per_order.items():
            stats[key]=dict(fixed_budgets={str(t):assess(diagnose(d['scores'][:,t]),np.full(80,t),target,families) for t in range(4)},stopping=assess(d['pred'],d['checks'],target,families))
        selected=dict(stopping=assess(selected_pred,selected_checks,target,families),full=assess(diagnose(selected_scores[:,3]),np.full(80,3),target,families),orders=[d['chosen'] for d in details])
        result[name]=dict(selected=selected,per_order=stats)
        predictions[name]=dict(scores=selected_scores.tolist(),predictions=selected_pred.tolist(),checks=selected_checks.tolist(),orders=selected_orders,folds=details,per_order={key:{k:v.tolist() for k,v in d.items()} for key,d in per_order.items()})
    l=result['latent']['selected'];r=result['raw']['selected'];s=l['stopping']
    gate=bool(s['gate_passed'] and s['correct']>=l['full']['correct']-1 and 3-s['mean_checks']>=.5 and r['stopping']['mean_checks']-s['mean_checks']>=.5)
    result['development_gate_passed']=gate;result['live_confirmation_warranted']=gate
    (out/'summary.json').write_text(json.dumps(result,indent=2));(out/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),families=families.tolist(),methods=predictions)))
    (out/'splits.json').write_text(json.dumps(splits,indent=2));(out/'inputs.json').write_text(json.dumps(hashes,indent=2))
    assert hashes['development_sha256']==hashlib.sha256(source.read_bytes()).hexdigest() and hashes['features_sha256']==hashlib.sha256(feature_path.read_bytes()).hexdigest()
    print('ORDER_COMPARISON_COMPLETE',gate,flush=True)
if __name__=='__main__':main()
