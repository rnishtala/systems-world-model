"""Fresh acquisition using the pilot's unchanged measurement functions."""
import json,sys,time
from pathlib import Path
from microservice import live
# The pilot collector also runs as a standalone ConfigMap script.
sys.modules.setdefault('live',live)
from microservice.ledger_probe import RNG,batch,frame

def collect(path):
    rng=RNG(9202);cases=[(f,j) for f in range(8) for j in range(4)];rng.r.shuffle(cases)
    rows=[];live.wait_ready()
    try:
        with Path(path).open('x') as out:
            for episode,(fault,rep) in enumerate(cases):
                start=time.monotonic();live.configure(fault,rng)
                for target in ('checkout','ledger'):
                    for _ in range(2):
                        if batch(target,9)!=9:raise RuntimeError('Warmup failed')
                loads=[3,9] if rep%2==0 else [9,3]
                base=[frame('checkout',n) for n in loads*2]
                order=['checkout','ledger'] if rep<2 else ['ledger','checkout']
                extra={target:[frame(target,n) for n in loads] for target in order}
                seconds=time.monotonic()-start
                row=dict(split='test',seed=9202,episode=episode,fault=fault,repeat=rep,base=base,extra=extra,order=order,seconds=seconds,utc=time.time())
                if seconds>35:raise RuntimeError('Incident exceeded lease safety budget')
                out.write(json.dumps(row)+'\n');out.flush();rows.append(row);live.reset()
                print('PROBE_TEST_COLLECT',episode+1,flush=True)
    finally:live.reset()
    return rows
