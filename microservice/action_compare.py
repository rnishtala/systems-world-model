"""Fresh, frozen-model outcome comparison; no training or policy selection."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from microservice import action_outcomes as a
from microservice.action_balanced import schedule
from microservice.action_forecast import Forecaster,verify_source,checkpoint_hash,PARTICLES,RNG_SEED

def fresh_cases(seed,previous):
    cases=[c for c in schedule(seed) if c['split']=='test']
    known={json.dumps(c['profile'],sort_keys=True) for c in previous}
    if any(json.dumps(c['profile'],sort_keys=True) in known for c in cases):raise ValueError('Source profile overlap')
    return cases

def paired_report(rows,baseline):
    groups=sorted({r['group'] for r in rows})
    differences=np.array([np.mean([abs(r['predictions']['dreamer'][0]-r['outcome'][0])-abs(r['predictions'][baseline][0]-r['outcome'][0]) for r in rows if r['group']==g]) for g in groups])
    rng=np.random.default_rng(777)
    boot=rng.choice(differences,(10000,len(groups)),replace=True).mean(1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(baseline=baseline,dreamer_minus_baseline=float(differences.mean()),bootstrap_95=[float(lo),float(hi)],profiles=len(groups),wins=int((differences<0).sum()),substantial_improvement=bool(differences.mean()<=-.05 and hi<0))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','out'):p.add_argument('--'+k,required=True)
    p.add_argument('--fresh-residual',action='store_true');p.add_argument('--fresh-readout',action='store_true');p.add_argument('--seed',type=int,required=True);args=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    source=Path(args.source);meta=verify_source(source);before=checkpoint_hash(source)
    # Resolve only the original balanced source tied to this checkpoint.
    uid=meta['source_run']
    from launcher.common import validate_retest
    validate_retest(source.parent.name,uid)
    original=source.parent.parent/uid/'temporal-pilot'
    protocol=json.loads((original/'protocol.json').read_text())
    summary=json.loads((original/'summary.json').read_text())['action_outcomes']
    raw=(original/'frozen-baselines.json').read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if not protocol.get('balanced') or not summary['all_restored'] or digest!=summary['frozen_sha256']:raise ValueError('Invalid baseline source')
    if hashlib.sha256((original/'examples.json').read_bytes()).hexdigest()!=meta['source_examples_sha256']:raise ValueError('Source examples changed')
    if args.seed==protocol['seed']:raise ValueError('Fresh seed required')
    frozen=json.loads(raw);cases=fresh_cases(args.seed,protocol['cases'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=dict(objective=meta.get('objective','original'),seed=args.seed,cases=cases,checkpoint_sha256=before,baseline_sha256=digest,training_sha256=meta['training_sha256'],source_run=source.parent.name,particles=PARTICLES,rng_seed=RNG_SEED,primary='Error-fraction MAE',selected=frozen['selected'],success='At least 0.05 absolute MAE reduction versus training-selected baseline; paired profile bootstrap upper bound below zero',timing='Dreamer inference offline using saved pre-action prefix only; no refitting',limits='10 independent profiles; exploratory bootstrap; no root-cause or repair-policy claim')
    if args.fresh_readout:
        from microservice.action_fresh_readout import freeze, score
        if meta.get('objective') != 'valid-channel-mask':
            raise ValueError('Expected full masked checkpoint')
        model = Forecaster(source)
        plan['readouts'] = freeze(model, source, out, residual=args.fresh_residual)
        plan.update(primary=plan['readouts']['primary'], selected='snapshot_ridge',
                    success=plan['readouts']['decision'])
    (out/'protocol.json').write_text(json.dumps(plan,indent=2));(out/'frozen-baselines.json').write_bytes(raw)
    if not args.fresh_readout:model=Forecaster(source)
    # Compile before fault injection. No outcomes are passed to the predictor.
    model(np.zeros((4,len(a.FEATURES))),a.ACTIONS[0])
    rows=[]
    for i,case in enumerate(cases):
        row={k:case[k] for k in ('split','group','action','regime')}
        def before_action(windows):
            row['observations']=a.observations(windows);row['predictions']=a.predictions(frozen,row)
            (out/f'prediction-{i}.json').write_text(json.dumps(row))
        for attempt in range(3):
            result=a.incident(case['mechanism'],case['action'],case['profile'],before_action=before_action)
            if not result['restored']:raise ValueError('Incident restoration failed')
            try:
                target=a.outcome(result['windows'])
            except ValueError as exc:
                if str(exc) != 'Insufficient outcome windows':
                    raise
                (out/f'incomplete-{i}-{attempt}.json').write_text(json.dumps(result))
                if attempt == 2:
                    raise
                continue
            (out/f'incident-{i}.json').write_text(json.dumps(result))
            row.update(outcome=target,restored=True)
            rows.append(row)
            break
        print('ACTION_COMPARISON_PROGRESS',i+1,'/'+str(len(cases)),flush=True)
    for i,row in enumerate(rows):
        row['predictions']['dreamer']=model(row['observations'],row['action'])
        (out/f'dreamer-prediction-{i}.json').write_text(json.dumps({'prediction':row['predictions']['dreamer']}))
    if checkpoint_hash(source)!=before or verify_source(source)['training_sha256']!=meta['training_sha256'] or hashlib.sha256((original/'frozen-baselines.json').read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
    scoring=dict(frozen,models=dict(frozen['models'],dreamer={}))
    scores=a.evaluate(rows,scoring)
    scores.update(paired=paired_report(rows,frozen['selected']),by_regime={regime:a.evaluate([r for r in rows if r['regime']==regime],scoring) for regime in sorted({r['regime'] for r in rows})},all_restored=True,checkpoint_unchanged=True,training_trials=0,checkpoint_sha256=before,frozen_sha256=digest)
    if args.fresh_readout:
        scores['readouts'] = score(model, rows, out, plan['readouts'])
        if checkpoint_hash(source) != before:
            raise ValueError('Checkpoint changed during readout scoring')
    (out/'examples.json').write_text(json.dumps(rows))
    (out/'summary.json').write_text(json.dumps(dict(action_comparison=scores,incidents=[]),indent=2))
    print('ACTION_COMPARISON_COMPLETE',flush=True)

if __name__=='__main__':main()
