"""Collect leased faults inside the isolated lab namespace."""
import argparse,json,os,time
from pathlib import Path
from microservice import live
from microservice.ledger_probe import RNG,batch,frame

def training_cases(seed):
    rng=RNG(seed);cases=[]
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

def probe_order(repeat, schedule='mixed'):
    if schedule not in ('mixed','ledger-first'):raise ValueError('Unknown probe schedule')
    return ['ledger','checkout'] if schedule=='ledger-first' or repeat%4>=2 else ['checkout','ledger']

def collect(path,training,seed,schedule='mixed'):
    probe_order(0,schedule)  # Reject invalid schedules before touching the lab.
    rng=RNG(seed)
    cases=training_cases(seed) if training else [dict(regime=g,fault=f,repeat=r) for g in (0,1) for f in range(8) for r in range(4)]
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
                    order=probe_order(c['repeat'],schedule)
                    base=[frame('checkout',n) for n in loads*2];extra={target:[frame(target,n) for n in loads] for target in order}
                    seconds=time.monotonic()-start
                    row=dict(seed=seed,episode=i,fault=c['fault'],repeat=c['repeat'],regime=c['regime'],base=base,extra=extra,order=order,seconds=seconds,utc=time.time())
                    if seconds>35:raise RuntimeError('Incident exceeded lease safety budget')
                    out.write(json.dumps(row)+'\n');out.flush();rows.append(row)
                finally:live.reset()
                print('PROBE_RANGE_COLLECT',('train' if training else 'test'),i+1,'/64',flush=True)
    finally:live.reset()
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--probe-schedule',choices=['mixed','ledger-first'],default='mixed');a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':
        p.error('Run inside systems-world-model-lab with POD_NAMESPACE from the downward API')
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    collect(a.out,a.split=='train',a.seed,a.probe_schedule)
if __name__=='__main__':main()
