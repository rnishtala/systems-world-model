"""Bounded acquisition pilot. Standard library only; no model/PVC access."""
import concurrent.futures,json,random,sys,time
from microservice import live

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
