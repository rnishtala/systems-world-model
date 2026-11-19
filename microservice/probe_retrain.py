"""Retrain upstream Dreamer on designated expanded training data; untouched confirmation."""
from microservice.probe_range import *

def collect(path,training):
    seed=9501 if training else 9503;rng=RNG(seed)
    cases=training_cases() if training else [dict(regime=g,fault=f,repeat=r) for g in (0,1) for f in range(8) for r in range(4)]
    if not training:rng.r.shuffle(cases)
    rows=[];live.wait_ready()
    try:
        with Path(path).open('x') as out:
            for i,c in enumerate(cases):
                start=time.monotonic()
                try:
                    if training:configure_training(c)
                    else:live.configure(c['fault'],rng,bool(c['regime']))
                    for target in ('checkout','ledger'):
                        for _ in range(2):
                            if batch(target,9)!=9:raise RuntimeError('Warmup failed')
                    loads=[3,9] if c['regime']==0 else [4,10]
                    if c['repeat']%2:loads.reverse()
                    order=['checkout','ledger'] if c['repeat']%4<2 else ['ledger','checkout']
                    base=[frame('checkout',n) for n in loads*2];extra={target:[frame(target,n) for n in loads] for target in order}
                    seconds=time.monotonic()-start
                    row=dict(seed=seed,episode=i,fault=c['fault'],repeat=c['repeat'],regime=c['regime'],base=base,extra=extra,order=order,seconds=seconds,utc=time.time())
                    if seconds>35:raise RuntimeError('Incident exceeded lease safety budget')
                    out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
                finally:live.reset()
                print('PROBE_RANGE_COLLECT',('train' if training else 'test'),i+1,'/64',flush=True)
    finally:live.reset()
    return rows

def evaluate(xs,rows,models,old,refit,old_x):
    y=labels(rows);truth=diagnose(y)
    features={f'{n}{t+1}':xs[f'{n}_{t}'] for n in ('latent','history') for t in (5,7)}
    pred={n:diagnose(score(m,features[n])) for n,m in models.items()};pred['old_latent6']=diagnose(score(old,old_x))
    pred['refit_latent6']=diagnose(score(refit,old_x))
    methods={n:stats(p,rows) for n,p in pred.items()};regimes={}
    for g in (0,1):
        mask=np.array([r['regime']==g for r in rows]);rs=[r for r in rows if r['regime']==g]
        regimes[str(g)]={n:stats(p[mask],rs) for n,p in pred.items()}
    bit_errors={n:np.sum(((p[:,None]>>np.arange(3))&1)!=(y>0),axis=0).tolist() for n,p in pred.items()}
    false_pool={n:int(sum(((p&2)>0)&((truth&2)==0))) for n,p in pred.items()}
    paired={};a=pred['latent6']
    for n,b in pred.items():
        if n=='latent6':continue
        wins=int(sum((a==truth)&(b!=truth)));losses=int(sum((a!=truth)&(b==truth)));total=wins+losses
        paired[n]=dict(new_only_correct=wins,control_only_correct=losses,p=min(1.,2*sum(math.comb(total,k) for k in range(min(wins,losses)+1))/2**total) if total else 1.)
    l=methods['latent6'];reliable=l['correct']>=58 and l['families']['0']['correct']==8 and all(v['correct']>=6 for v in l['families'].values())
    rg=all(m['latent6']['correct']>=29 and m['latent6']['families']['0']['correct']==4 and all(v['correct']>=3 for v in m['latent6']['families'].values()) for m in regimes.values())
    improved=l['correct']>=methods['refit_latent6']['correct']+2 and false_pool['latent6']<false_pool['refit_latent6']
    matched=l['correct']>=max(methods[n]['correct'] for n in ('history6','latent8'))-2
    result=dict(methods=methods,regimes=regimes,bit_errors=bit_errors,false_pool=false_pool,paired=paired,reliable=bool(reliable),regime_gate=bool(rg),improvement_gate=bool(improved),matched_controls=bool(matched),feasibility_gate=bool(reliable and rg and improved and matched),world_model_retrained=True)
    return result,dict(target=truth.tolist(),methods={n:p.tolist() for n,p in pred.items()})


def load_new():
    config=json.loads(Path('evaluation/frozen.json').read_text())
    archive=np.load('evaluation/readouts.npz',allow_pickle=False)
    assert hashlib.sha256(Path('evaluation/readouts.npz').read_bytes()).hexdigest()==config['readouts_sha256']
    return config,{n:dict(config=c['config'],**{k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith(n+'__')}) for n,c in config['configs'].items()}

def main():
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    phase=sys.argv[1];out=Path('evaluation');root=Path('models')
    rows=[json.loads(l) for l in Path('collection/train.jsonl').read_text().splitlines()]
    assert len(rows)==128 and {r['seed'] for r in rows}=={9101,9103,9501}
    if phase=='fit':
        out.mkdir(exist_ok=False);model=Model(str(root),101,'native')
        (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
        xs=extract_prefixes(model,rows);np.savez_compressed(out/'train-features.npz',**{f'{n}_{t}':xs[f'{n}_{t}'] for n in ('latent','history') for t in (5,7)})
        y=labels(rows);groups=np.array([r['repeat']%4 for r in rows]);models={};configs={}
        for n in ('latent','history'):
            for t in (5,7):
                name=f'{n}{t+1}';config,counts=choose(xs[f'{n}_{t}'],y,groups);models[name]=fit(xs[f'{n}_{t}'],y,config);configs[name]=dict(config=config,cv_correct=counts)
        np.savez_compressed(out/'readouts.npz',**{f'{n}__{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
        frozen=dict(utc=time.time(),configs=configs,checkpoint_sha256=digest(root/'final-101'),readouts_sha256=hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest(),train_sha256=hashlib.sha256(Path('collection/train.jsonl').read_bytes()).hexdigest())
        (out/'frozen.json').write_text(json.dumps(frozen,indent=2));print('PROBE_RETRAIN_FROZEN',flush=True)
    elif phase=='test':
        frozen,_=load_new();assert digest(root/'final-101')==frozen['checkpoint_sha256']
        test=collect(out/'test.jsonl',False);model=Model(str(root),101,'native');xs=extract_prefixes(model,test)
        np.savez_compressed(out/'test-features.npz',**{f'{n}_{t}':xs[f'{n}_{t}'] for n in ('latent','history') for t in (5,7)})
        assert digest(root/'final-101')==frozen['checkpoint_sha256']
    elif phase=='baseline':
        frozen,models=load_new();previous,old=load_readouts(Path('frozen'))
        oldroot=Path('/results/probe-target-model-v1/models');assert digest(oldroot/'final-91')==previous['checkpoint_sha256']
        config=json.loads(Path('frozen/refit-config.json').read_text());archive=np.load('frozen/refit-readouts.npz',allow_pickle=False)
        assert hashlib.sha256(Path('frozen/refit-readouts.npz').read_bytes()).hexdigest()==config['readouts_sha256']
        refit=dict(config=config['configs']['latent6']['config'],**{k.split('__',1)[1]:archive[k] for k in archive.files if k.startswith('latent6__')})
        test=[json.loads(l) for l in (out/'test.jsonl').read_text().splitlines()];assert len(test)==64 and {r['seed'] for r in test}=={9503}
        assert frozen['utc']<min(r['utc'] for r in test)
        model=Model(str(oldroot),91,'native');ox=extract_prefixes(model,test)
        np.savez_compressed(out/'baseline-features.npz',latent_5=ox['latent_5'])
        xs=np.load(out/'test-features.npz',allow_pickle=False)
        result,pred=evaluate(xs,test,models,old['latent6'],refit,ox['latent_5'])
        assert digest(oldroot/'final-91')==previous['checkpoint_sha256'] and digest(root/'final-101')==frozen['checkpoint_sha256']
        result.update(checkpoints_unchanged_after_freeze=True,actual_early_stopping=False)
        for n,v in [('summary',result),('predictions',pred)]:(out/f'{n}.json').write_text(json.dumps(v,indent=2)+'\n')
        print('PROBE_RETRAIN_COMPLETE',json.dumps(result),flush=True)
    else:raise ValueError(phase)
if __name__=='__main__':main()
