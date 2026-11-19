"""Bounded acquisition pilot. Standard library only; no model/PVC access."""
import concurrent.futures,json,random,sys,time
import live

class RNG:
    def __init__(self,seed):self.r=random.Random(seed)
    def uniform(self,a,b):return self.r.uniform(a,b)
    def integers(self,a,b):return self.r.randrange(a,b)

def batch(target,n):
    def one(_):
        try:
            result=live.request(target,'/request')
            return bool(result['trace']) and not any(t['error'] for t in result['trace'])
        except Exception:return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        return sum(ex.map(one,range(n)))

def frame(target,n):
    live.reset_measurement_window();start=time.monotonic();ok=batch(target,n)
    metrics={}
    for s in live.SERVICES:
        m=live.request(s,'/metrics')
        metrics[s]={k:m[k] for k in ('count','latency_p95','self_p95','downstream_p95','error_rate')}
        expected=n if target=='checkout' or s=='ledger' else 0
        if m['count']!=expected or m['error_rate']!=0:raise RuntimeError('Invalid measurement counts/errors')
    if ok!=n:raise RuntimeError('Failed measured requests')
    return dict(target=target,requests=n,successes=ok,metrics=metrics,seconds=time.monotonic()-start)

def main():
    split=sys.argv[1];assert split in ('development','test')
    seed=9101 if split=='development' else 9103;rng=RNG(seed)
    cases=[(fault,rep) for fault in range(8) for rep in range(4)];rng.r.shuffle(cases)
    live.wait_ready();print('START',json.dumps(dict(split=split,seed=seed,utc=time.time())),flush=True)
    try:
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
            row=dict(split=split,seed=seed,episode=episode,fault=fault,repeat=rep,
                     base=base,extra=extra,order=order,seconds=seconds,utc=time.time())
            if seconds>35:raise RuntimeError('Incident exceeded lease safety budget')
            print('ROW',json.dumps(row),flush=True)
            live.reset()
    finally:
        live.reset();print('RESET_COMPLETE',flush=True)
    print('COMPLETE',split,flush=True)
if __name__=='__main__':main()
