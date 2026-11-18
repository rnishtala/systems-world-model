"""Fresh training-only lab collection with clean measurement windows."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from microservice.live import configure,frame,reset,wait_ready,batch,reset_measurement_window

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(8101)
    cases=[(f,bool(j%2)) for f in range(8) for j in range(10)];rng.shuffle(cases);wait_ready()
    try:
        with (out/'train.jsonl').open('x') as stream:
            for i,(fault,shift) in enumerate(cases):
                start=time.monotonic();n=configure(fault,rng,shift)
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup failed')
                reset_measurement_window();frames=[frame(n) for _ in range(4)]
                row=dict(episode=i,fault=fault,seed=8101,shifted=shift,measurement_window='post-warmup',frames=frames,seconds=time.monotonic()-start)
                stream.write(json.dumps(row)+'\n');stream.flush()
                if row['seconds']>35 or any(f['successes']!=f['requests'] for f in frames):raise RuntimeError('Invalid incident, retained')
                if any(m['count']!=(j+1)*n for j,f in enumerate(frames) for m in f['metrics'].values()):raise RuntimeError('Contaminated measurement window')
                print('FRESH_TRAIN_COLLECT',i+1,fault,flush=True)
    finally:reset()
    (out/'identity.json').write_text(json.dumps(dict(seed=8101,n=80,sha256=hashlib.sha256((out/'train.jsonl').read_bytes()).hexdigest(),measurement_window='post-warmup'),indent=2))
if __name__=='__main__':main()
