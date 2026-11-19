"""New-range supervised readouts on frozen Dreamer states; fresh confirmation."""
import json,time,hashlib,sys,math
from pathlib import Path
import numpy as np
from microservice import live
sys.modules.setdefault('live',live)
from microservice.ledger_probe import RNG,batch,frame
from microservice.probe_target import frames
from microservice.probe_prefix import extract_prefixes
from microservice.probe_six import load_readouts,stats
from microservice.partial_readout import fit,score,diagnose
from microservice.latent_probe import labels,digest
from scripts.develop_forecast_readout import choose

def training_cases():
    rng=RNG(9501);cases=[]
    for fault in range(8):
        bands=[]
        for _ in range(4):
            v=[0]*4+[1]*4;rng.r.shuffle(v);bands.append(v)
        for rep in range(8):
            cpu=rng.uniform(*((.025,.05) if bands[0][rep]==0 else (.055,.075)))
            delay=rng.uniform(*((.12,.23) if bands[1][rep]==0 else (.26,.34)))
            hold=rng.uniform(*((.025,.035) if bands[2][rep]==0 else (.045,.055)))
            cases.append(dict(fault=fault,repeat=rep,regime=rep//4,cpu=cpu,delay=delay,hold=hold,pool=1+bands[3][rep]))
    rng.r.shuffle(cases);return cases

def configure_training(c):
    live.reset();fault=c['fault'];control=dict(db_hold=c['hold'])
    if fault in (2,5,6,7):control['pool_size']=c['pool']
    if fault in (3,4,6,7):control['delay_seconds']=c['delay']
    live.request('ledger','/control',control,True)
    if fault in (1,4,5,7):live.request('payment','/control',dict(cpu_seconds=c['cpu']),True)

def collect(path,training):
    seed=9501 if training else 9502;rng=RNG(seed)
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

def evaluate(xs,rows,models,old):
    y=labels(rows);truth=diagnose(y)
    features={f'{n}{t+1}':xs[f'{n}_{t}'] for n in ('latent','history') for t in (5,7)}
    pred={n:diagnose(score(m,features[n])) for n,m in models.items()};pred['old_latent6']=diagnose(score(old,xs['latent_5']))
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
    improved=l['correct']>=methods['old_latent6']['correct']+2 and false_pool['latent6']<false_pool['old_latent6']
    matched=l['correct']>=max(methods[n]['correct'] for n in ('history6','latent8'))-2
    result=dict(methods=methods,regimes=regimes,bit_errors=bit_errors,false_pool=false_pool,paired=paired,reliable=bool(reliable),regime_gate=bool(rg),improvement_gate=bool(improved),matched_controls=bool(matched),feasibility_gate=bool(reliable and rg and improved and matched),world_model_retrained=False)
    return result,dict(target=truth.tolist(),methods={n:p.tolist() for n,p in pred.items()})

def main():
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    root=Path('/results/probe-target-model-v1/models');previous,oldmodels=load_readouts(Path('frozen'))
    assert digest(root/'final-91')==previous['checkpoint_sha256']
    out=Path('evaluation');out.mkdir(exist_ok=False)
    # Training acquisition is independent of models; labels/controls never enter features.
    new=collect(Path('collection/new-train.jsonl'),True)
    prior=[json.loads(l) for l in Path('collection/train.jsonl').read_text().splitlines()]
    assert len(prior)==64 and {r['seed'] for r in prior}=={9101,9103}
    rows=prior+new;model=Model(str(root),91,'native');(out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    xs=extract_prefixes(model,rows);np.savez_compressed(out/'train-features.npz',**{f'{n}_{t}':xs[f'{n}_{t}'] for n in ('latent','history') for t in (5,7)})
    y=labels(rows);groups=np.array([r['repeat']%4 for r in rows]);models={};configs={}
    for n in ('latent','history'):
        for t in (5,7):
            name=f'{n}{t+1}';config,counts=choose(xs[f'{n}_{t}'],y,groups);models[name]=fit(xs[f'{n}_{t}'],y,config);configs[name]=dict(config=config,cv_correct=counts)
    np.savez_compressed(out/'readouts.npz',**{f'{n}__{k}':v for n,m in models.items() for k,v in m.items() if k!='config'})
    frozen=dict(utc=time.time(),configs=configs,checkpoint_sha256=previous['checkpoint_sha256'],readouts_sha256=hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest(),train_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path('collection/train.jsonl'),Path('collection/new-train.jsonl'))})
    (out/'frozen.json').write_text(json.dumps(frozen,indent=2));print('PROBE_RANGE_READOUTS_FROZEN',flush=True)
    test=collect(out/'test.jsonl',False);tx=extract_prefixes(model,test)
    np.savez_compressed(out/'test-features.npz',**{f'{n}_{t}':tx[f'{n}_{t}'] for n in ('latent','history') for t in (5,7)})
    result,pred=evaluate(tx,test,models,oldmodels['latent6'])
    assert digest(root/'final-91')==previous['checkpoint_sha256'] and hashlib.sha256((out/'readouts.npz').read_bytes()).hexdigest()==frozen['readouts_sha256'];load_readouts(Path('frozen'))
    result.update(checkpoint_unchanged=True,readouts_unchanged=True)
    for n,v in [('summary',result),('predictions',pred)]:(out/f'{n}.json').write_text(json.dumps(v,indent=2)+'\n')
    print('PROBE_RANGE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
