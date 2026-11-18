"""Fresh independent metric windows; fault and load controls excluded from views."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from microservice.live import configure,frame,reset,wait_ready,batch,reset_measurement_window


def collect(path,test=False):
    seed=8602 if test else 8601;rng=np.random.default_rng(seed)
    cases=[(f,j%2) for f in range(8) for j in range(10)];rng.shuffle(cases)
    rows=[];wait_ready()
    try:
        with Path(path).open('x') as stream:
            for i,(fault,phase) in enumerate(cases):
                start=time.monotonic();configure(fault,rng,False)
                lo,hi=(7,9) if test else (4,6)
                loads=[hi,lo,hi,lo] if phase else [lo,hi,lo,hi]
                for _ in range(2):
                    if batch(loads[0])!=loads[0]:raise RuntimeError('Warmup failed')
                frames=[]
                for n in loads:
                    reset_measurement_window();frames.append(frame(n))
                row=dict(episode=i,seed=seed,fault=fault,phase=phase,loads=loads,
                         measurement_window='independent-batch',frames=frames,seconds=time.monotonic()-start)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if row['seconds']>35 or any(f['successes']!=f['requests'] for f in frames):raise RuntimeError('Invalid incident retained')
                if any(m['count']!=f['requests'] for f in frames for m in f['metrics'].values()):raise RuntimeError('Contaminated window retained')
                rows.append(row);print('TEMPORAL_COLLECT',seed,i+1,flush=True)
    finally:reset()
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rows=collect(out/'train.jsonl')
    (out/'identity.json').write_text(json.dumps(dict(seed=8601,n=len(rows),sha256=hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest()),indent=2))
