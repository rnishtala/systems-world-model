"""Collect fresh regime-separated measurements with bounded leased faults."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from microservice.live import configure,frame,reset,wait_ready,batch,reset_measurement_window
from microservice.regime import cases,loads,SEEDS

def collect(path,split):
    seed=SEEDS[split];rng=np.random.default_rng(seed);schedule=cases(split);rng.shuffle(schedule)
    rows=[];wait_ready()
    try:
        with Path(path).open('x') as stream:
            for i,(fault,regime,phase) in enumerate(schedule):
                start=time.monotonic();configure(fault,rng,False);traffic=loads(split,regime,phase)
                for _ in range(2):
                    if batch(traffic[0])!=traffic[0]:raise RuntimeError('Warmup failed')
                frames=[]
                for n in traffic:
                    reset_measurement_window();frames.append(frame(n))
                row=dict(episode=i,seed=seed,split=split,fault=fault,regime=regime,phase=phase,loads=traffic,
                         measurement_window='independent-batch',frames=frames,seconds=time.monotonic()-start)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if row['seconds']>35 or any(f['successes']!=f['requests'] for f in frames):raise RuntimeError('Invalid incident retained')
                if any(m['count']!=f['requests'] for f in frames for m in f['metrics'].values()):raise RuntimeError('Contaminated window retained')
                rows.append(row);print('REGIME_COLLECT',split,i+1,len(schedule),flush=True)
    finally:reset()
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rows=collect(out/'train.jsonl','train')
    (out/'identity.json').write_text(json.dumps(dict(seed=8701,n=len(rows),sha256=hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest()),indent=2))
