"""Fresh fixed-policy order confirmation; no fitting or selection after collection."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.order_features import extract
from microservice.latent_probe import digest
from microservice.efficiency import collect,decisions
from microservice.partial_readout import score,assess,diagnose
from microservice.mechanisms import FAULT_MASK

def load_readout(params,policy,name,t):
    return dict(config=policy['policies'][name]['readouts'][t],**{k:params[f'{name}_{t}_{k}'] for k in ['mean','scale','bias','z','dual']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--frozen',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);frozen=Path(a.frozen);policy_path=frozen/'policy.json';readout_path=frozen/'readouts.npz'
    ph=hashlib.sha256(policy_path.read_bytes()).hexdigest();rh=hashlib.sha256(readout_path.read_bytes()).hexdigest();policy=json.loads(policy_path.read_text());assert policy['live_eligible'] and policy['readouts_sha256']==rh and policy['scenario_seed']==8401
    checkpoint=Path(a.root)/'final-51';before=digest(checkpoint);assert before==policy['checkpoint_sha256']
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    model=Model(a.root,51,'inspect-mechanisms');audited=audit(model)
    params=np.load(readout_path);readouts={name:[load_readout(params,policy,name,t) for t in range(4)] for name in ['latent','raw']}
    (out/'before.json').write_text(json.dumps(dict(utc=time.time(),policy_sha256=ph,readouts_sha256=rh,checkpoint_sha256=before,label_audit=audited),indent=2))
    rows=collect(out/'test.jsonl',8401);features,actions=extract(model,rows);np.savez_compressed(out/'features.npz',**features)
    families=np.array([r['fault'] for r in rows]);target=np.array([FAULT_MASK[f] for f in families]);summary={};predictions={}
    for name in ['latent','raw']:
        order=policy['policies'][name]['order'];threshold=policy['policies'][name]['threshold']
        values=np.stack([score(readouts[name][t],features[f'{name}_{order}_{t}']) for t in range(4)],axis=1)
        pred,checks=decisions(values,threshold)
        summary[name]=dict(order=order,threshold=threshold,stopping=assess(pred,checks,target,families),full=assess(diagnose(values[:,3]),np.full(80,3),target,families),fixed_budgets={str(t):assess(diagnose(values[:,t]),np.full(80,t),target,families) for t in range(4)})
        predictions[name]=dict(scores=values.tolist(),predictions=pred.tolist(),checks=checks.tolist())
    l=summary['latent'];r=summary['raw'];s=l['stopping']
    summary['confirmation_passed']=bool(s['gate_passed'] and s['correct']>=l['full']['correct']-1 and 3-s['mean_checks']>=.5 and r['stopping']['mean_checks']-s['mean_checks']>=.5)
    lp=np.array(predictions['latent']['predictions']);rp=np.array(predictions['raw']['predictions'])
    summary['paired_latent_only_correct']=int(sum((lp==target)&(rp!=target)));summary['paired_raw_only_correct']=int(sum((rp==target)&(lp!=target)))
    assert digest(checkpoint)==before and hashlib.sha256(policy_path.read_bytes()).hexdigest()==ph and hashlib.sha256(readout_path.read_bytes()).hexdigest()==rh
    assert actions==1200
    summary['checkpoint_and_policy_unchanged']=True;summary['verified_previous_actions']=actions
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'predictions.json').write_text(json.dumps(dict(target=target.tolist(),families=families.tolist(),methods=predictions)))
    print('ORDER_CONFIRMATION_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
