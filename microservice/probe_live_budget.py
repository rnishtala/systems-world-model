"""Bounded actual acquisition budgets with frozen, label-free inference."""
import json,sys,time,hashlib,math
from pathlib import Path
import numpy as np
from microservice import live
sys.modules.setdefault('live',live)
from microservice.ledger_probe import RNG,batch,frame
from microservice.budget_features import extract_budget
from microservice.probe_six import load_readouts,stats
from microservice.partial_readout import score,diagnose
from microservice.latent_probe import digest,labels

def schedule(regime,rep):
    loads=[3,9] if regime==0 else [4,10]
    if rep%2:loads.reverse()
    order=['checkout','ledger'] if rep<2 else ['ledger','checkout']
    return [['checkout',n] for n in loads*2]+[[target,n] for target in order for n in loads]

def assess(rows):
    methods={};by={b:sorted([r for r in rows if r['budget']==b],key=lambda r:r['pair']) for b in (6,8)}
    assert len(by[6])==len(by[8])==64
    for a,b in zip(by[6],by[8]):assert (a['pair'],a['fault'],a['regime'],a['scenario_seed'])==(b['pair'],b['fault'],b['regime'],b['scenario_seed'])
    for budget,rs in by.items():
        for name in ('latent','history','summary_history'):
            pred=np.array([r['predictions'][name] for r in rs]);methods[f'{name}{budget}']=stats(pred,rs)
    regimes={}
    for regime in (0,1):
        rs=[r for r in by[6] if r['regime']==regime];regimes[str(regime)]=stats(np.array([r['predictions']['latent'] for r in rs]),rs)
    target=diagnose(labels(by[6]));a=np.array([r['predictions']['latent'] for r in by[6]]);paired={}
    for budget,rs in by.items():
        for name in ('latent','history','summary_history'):
            if budget==6 and name=='latent':continue
            b=np.array([r['predictions'][name] for r in rs]);wins=int(sum((a==target)&(b!=target)));losses=int(sum((a!=target)&(b==target)));n=wins+losses
            paired[f'{name}{budget}']=dict(latent6_only_correct=wins,control_only_correct=losses,p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.)
    costs={}
    for budget,rs in by.items():
        costs[str(budget)]={key:float(np.mean([r[key] for r in rs])) for key in ('measured_requests','total_requests','acquisition_seconds','warmup_inclusive_seconds','inference_seconds')}
        costs[str(budget)]['model_inclusive_seconds']=costs[str(budget)]['warmup_inclusive_seconds']+costs[str(budget)]['inference_seconds']
    savings={key:1-costs['6'][key]/costs['8'][key] for key in costs['6']}
    l=methods['latent6'];reliable=l['correct']>=58 and l['families']['0']['correct']==8 and all(v['correct']>=6 for v in l['families'].values())
    regime_gate=all(m['correct']>=29 and m['families']['0']['correct']==4 and all(v['correct']>=3 for v in m['families'].values()) for m in regimes.values())
    matched=l['correct']>=max(methods[n]['correct'] for n in ('latent8','history8','summary_history8'))-2
    efficiency=savings['measured_requests']>=.2 and savings['warmup_inclusive_seconds']>0
    return dict(methods=methods,regimes=regimes,paired=paired,mean_costs=costs,fraction_saved=savings,reliable=bool(reliable),regime_gate=bool(regime_gate),within_two_of_full_controls=bool(matched),efficiency_gate=bool(efficiency),feasibility_gate=bool(reliable and regime_gate and matched and efficiency),actual_early_stopping=True)

def main():
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    frozen,models=load_readouts(Path('frozen'));root=Path('/results/probe-target-model-v1/models')
    assert digest(root/'final-91')==frozen['checkpoint_sha256']
    out=Path('evaluation');out.mkdir(exist_ok=False);model=Model(str(root),91,'native')
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    # Compile both sizes on old training evidence before any fault/latency measurement.
    from microservice.probe_target import frames
    old=json.loads(Path('collection/train.jsonl').read_text().splitlines()[0]);fs=frames(old)
    for budget in (6,8):extract_budget(model,[dict(frames=fs[:budget],schedule=[[f['target'],f['requests']] for f in fs])])
    rng=RNG(9401);cases=[(g,f,r) for g in (0,1) for f in range(8) for r in range(4)];rng.r.shuffle(cases)
    rows=[];saved={6:[],8:[]};live.wait_ready();print('PROBE_BUDGET_FROZEN_VERIFIED',flush=True)
    try:
        with (out/'incidents.jsonl').open('x') as stream:
            for pair,(regime,fault,rep) in enumerate(cases):
                seed=940100+pair;plan=schedule(regime,rep)
                for budget in ([6,8] if pair%2==0 else [8,6]):
                    start=time.monotonic()
                    try:
                        live.configure(fault,RNG(seed),bool(regime))
                        for target in ('checkout','ledger'):
                            for _ in range(2):
                                if batch(target,9)!=9:raise RuntimeError('Warmup failed')
                        acquisition=time.monotonic();observed=[frame(target,n) for target,n in plan[:budget]]
                        acquisition_seconds=time.monotonic()-acquisition;elapsed=time.monotonic()-start
                        if elapsed>35:raise RuntimeError('Incident exceeded safety budget')
                        row=dict(pair=pair,regime=regime,fault=fault,repeat=rep,scenario_seed=seed,budget=budget,schedule=plan,frames=observed,utc=time.time(),
                            measured_requests=sum(f['requests'] for f in observed),total_requests=36+sum(f['requests'] for f in observed),acquisition_seconds=acquisition_seconds,warmup_inclusive_seconds=elapsed)
                    finally:live.reset()
                    begin=time.monotonic();xs=extract_budget(model,[row]);last={n:xs[f'{n}_{budget-1}'][0] for n in ('latent','history','summary_history')}
                    row['predictions']={n:int(diagnose(score(models[f'{n}{budget}'],x[None]))[0]) for n,x in last.items()}
                    row['inference_seconds']=time.monotonic()-begin
                    saved[budget].append(last);rows.append(row);stream.write(json.dumps(row)+'\n');stream.flush()
                    print('PROBE_BUDGET_INCIDENT',len(rows),'/128',flush=True)
        np.savez_compressed(out/'features.npz',**{f'{n}{b}':np.array([x[n] for x in xs]) for b,xs in saved.items() for n in ('latent','history','summary_history')})
        assert digest(root/'final-91')==frozen['checkpoint_sha256'];load_readouts(Path('frozen'))
        result=assess(rows);result.update(checkpoint_unchanged=True,readouts_unchanged=True)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PROBE_BUDGET_COMPLETE',json.dumps(result),flush=True)
    finally:live.reset()
if __name__=='__main__':main()
