"""Fresh lab comparison; freeze all readouts before new evaluation collection."""
from microservice.latent_probe import *
from microservice.latent_probe import extract as extract_state
def extract(model,rows):
    last,latent,diagnoses=extract_state(model,rows)
    history=np.array([np.asarray(observations(row)).reshape(-1) for row in rows])
    return history,latent,diagnoses
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--calibration',required=True);p.add_argument('--out',required=True);p.add_argument('--scenario-seed',type=int,default=6500);p.add_argument('--model-seed',type=int,default=21);p.add_argument('--measurement-window',choices=('legacy','post-warmup'),default='legacy');a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    from microservice.agent import Model
    from microservice.auxiliary_audit import audit
    before=digest(Path(a.root)/f'final-{a.model_seed}');model=Model(a.root,a.model_seed,'inspect-mechanisms')
    (out/'audit.json').write_text(json.dumps(audit(model),indent=2))
    train=list(map(json.loads,Path(a.calibration).read_text().splitlines()))
    raw,latent,training_predictions=extract(model,train);y=labels(train)
    probes={'raw':fit_probe(raw,y),'latent':fit_probe(latent,y),'shuffled_latent':fit_probe(latent,y[np.random.default_rng(7001).permutation(len(y))])}
    last_train=np.array([observations(row)[-1][2:10] for row in train]);probes['raw_last']=fit_probe(last_train,y)
    np.savez_compressed(out/'train-features.npz',raw=raw,latent=latent,labels=y)
    np.savez_compressed(out/'probes.npz',**{name+'_'+k:v for name,probe in probes.items() for k,v in probe.items()})
    (out/'fitted.json').write_text(json.dumps({'measurement_window':a.measurement_window,'utc':time.time(),'calibration_sha256':hashlib.sha256(Path(a.calibration).read_bytes()).hexdigest(),'checkpoint_sha256':before,'raw_dimensions':raw.shape[1],'latent_dimensions':latent.shape[1],'penalty':1.,'threshold':0.,'seed':a.scenario_seed,'scope':'Full-history raw and latent probes frozen before collecting fresh test incidents.'},indent=2))
    (out/'training-diagnosis.json').write_text(json.dumps({'correct':int(sum(training_predictions==np.array([FAULT_MASK[r['fault']] for r in train]))),'n':len(train)},indent=2))
    print('PROBES_FROZEN',flush=True)
    from microservice.live import configure,frame,reset,wait_ready,batch,reset_measurement_window
    rng=np.random.default_rng(a.scenario_seed);faults=np.tile(np.arange(8),10);rng.shuffle(faults);test=[];wait_ready()
    try:
        with (out/'test.jsonl').open('x') as f:
            for i,fault in enumerate(faults):
                start=time.monotonic();n=configure(int(fault),rng,True)
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup failed')
                if a.measurement_window=='post-warmup':reset_measurement_window()
                frames=[frame(n) for _ in range(4)]
                row=dict(measurement_window=a.measurement_window,episode=i,fault=int(fault),seed=a.scenario_seed,shifted=True,frames=frames,seconds=time.monotonic()-start)
                f.write(json.dumps(row)+'\n');f.flush()
                if row['seconds']>35 or any(x['successes']!=x['requests'] for x in frames):raise RuntimeError('Invalid incident, retained')
                if any(m['count']!=(j+1)*n for j,x in enumerate(frames) for m in x['metrics'].values()):raise RuntimeError('Contaminated measurement window, retained')
                test.append(row);print('COLLECT',i,int(fault),flush=True)
    finally:reset()
    raw,latent,original=extract(model,test)
    np.savez_compressed(out/'test-features.npz',raw=raw,latent=latent,labels=labels(test))
    predictions={'dreamer_auxiliary':original,**{name:predict(probe,raw if name=='raw' else np.array([observations(row)[-1][2:10] for row in test]) if name=='raw_last' else latent) for name,probe in probes.items()}}
    model.controller='inspect-first'
    _,_,reward_predictions=extract(model,test)
    predictions['dreamer_reward']=reward_predictions
    target=np.array([FAULT_MASK[r['fault']] for r in test]);summary={}
    for name,pred in predictions.items():
        families={str(f):{'n':10,'correct':int(sum(pred[i]==target[i] for i,r in enumerate(test) if r['fault']==f))} for f in range(8)}
        summary[name]={'correct':int(sum(pred==target)),'n':len(test),'families':families,'gate_passed':all(v['correct']>=(10 if f=='0' else 9) for f,v in families.items())}
    after=digest(Path(a.root)/f'final-{a.model_seed}');assert before==after,'Checkpoint changed'
    result={'methods':summary,'checkpoint_unchanged':True,'scope':'Fresh training and later evaluation batches; single model, known regimes only.', 'added_value_signal':summary['latent']['gate_passed'] and summary['latent']['correct']-summary['raw']['correct']>=4, 'paired_latent_wins':int(sum((predictions['latent']==target)&(predictions['raw']!=target))), 'paired_raw_wins':int(sum((predictions['raw']==target)&(predictions['latent']!=target)))}
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    (out/'predictions.json').write_text(json.dumps({'target':target.tolist(),**{k:v.tolist() for k,v in predictions.items()}},indent=2))
    print('PROBE_COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
