"""Collect actual HTTP/PostgreSQL trajectories; reset faults even on failure."""
import argparse,json,time
from pathlib import Path
import numpy as np
from microservice.live import configure,frame,reset,wait_ready
p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--per-class',type=int,default=20);p.add_argument('--seed',type=int,default=1200);p.add_argument('--shift',action='store_true');p.add_argument('--compositional',action='store_true');a=p.parse_args()
out=Path(a.out)
if out.exists():raise SystemExit('Refusing to replace evidence')
out.parent.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.seed)
faults=np.tile(np.arange(7 if a.compositional else 4),a.per_class);rng.shuffle(faults)
wait_ready()
try:
    with out.open('x') as f:
        for i,fault in enumerate(faults):
            start=time.time();shifted=bool(i%2) if a.compositional else a.shift;n=configure(int(fault),rng,shifted)
            frames=[frame(n) for _ in range(4)]
            if any(x['successes']!=x['requests'] for x in frames):raise RuntimeError('Invalid calibration: request failure')
            row=dict(episode=i,fault=int(fault),seed=a.seed,shifted=shifted,frames=frames,started=start,seconds=time.time()-start)
            f.write(json.dumps(row)+'\n');f.flush()
            print('COLLECT',i,int(fault),round(row['seconds'],2),flush=True)
finally:reset()
