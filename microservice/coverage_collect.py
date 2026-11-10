"""Add new triple-fault calibration; previous held-out incidents stay excluded."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
from microservice.live import configure,frame,reset,wait_ready,batch

def validate(rows,expected):
    if collections.Counter(r['fault'] for r in rows)!=collections.Counter(expected):raise ValueError('Unexpected family counts')
    if any(len(r['frames'])!=4 or any(f['successes']!=f['requests'] for f in r['frames']) for r in rows):raise ValueError('Invalid calibration frames')

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    original=Path(a.base).read_bytes()
    if hashlib.sha256(original).hexdigest()!='9121c40cab47960f8f41ea57b605fa20a41509b8c457abe60132ef07207f077b':raise ValueError('Original calibration hash mismatch')
    rows=[json.loads(l) for l in original.splitlines()];validate(rows,{i:20 for i in range(7)})
    rng=np.random.default_rng(6600);new=[];wait_ready()
    try:
        with (out/'triples.jsonl').open('x') as f:
            for i in range(20):
                start=time.monotonic();shift=bool(i%2);n=configure(7,rng,shift)
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup failed')
                frames=[frame(n) for _ in range(4)]
                if time.monotonic()-start>35:raise RuntimeError('Incident exceeded lease guard')
                row=dict(episode=i,fault=7,seed=6600,shifted=shift,frames=frames,seconds=time.monotonic()-start)
                validate([row],{7:1});new.append(row);f.write(json.dumps(row)+'\n');f.flush();print('CALIBRATION',i,flush=True)
    finally:reset()
    rows+=new;validate(rows,{i:20 for i in range(8)})
    data=original.rstrip()+b'\n'+(out/'triples.jsonl').read_bytes()
    (out/'calibration.jsonl').write_bytes(data)
    (out/'provenance.json').write_text(json.dumps(dict(base_sha256=hashlib.sha256(original).hexdigest(),calibration_sha256=hashlib.sha256(data).hexdigest(),base_incidents=140,new_incidents=20,seed=6600,scope='All eight regimes represented; no historical held-out data reused.'),indent=2))
    print('CALIBRATION_COMPLETE',flush=True)
if __name__=='__main__':main()
