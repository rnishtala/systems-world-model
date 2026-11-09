"""Experiment operator and telemetry adapter. The policy never gets controls/labels."""
import concurrent.futures,json,time,urllib.request
SERVICES=('checkout','payment','ledger')

def request(service,path,body=None,control=False):
    port=9090 if control else 8080
    req=urllib.request.Request(f'http://swm-{service}:{port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=12) as r:return json.load(r)

def reset():
    errors=[]
    for s in SERVICES:
        try:request(s,'/control',{},True)
        except Exception as e:errors.append(str(e))
    if errors:raise RuntimeError('Reset failed: '+'; '.join(errors))

def wait_ready(timeout=45):
    deadline=time.monotonic()+timeout
    while True:
        try:
            for s in SERVICES:request(s,'/health')
            reset();return
        except Exception:
            if time.monotonic()>=deadline:raise
            time.sleep(1)


def configure(fault,rng,shift=False):
    reset()
    hold=float(rng.uniform(.03,.05))
    request('ledger','/control',{'db_hold':hold},True)
    if fault==1:request('payment','/control',{'cpu_seconds':float(rng.uniform(.055,.075) if shift else rng.uniform(.025,.05))},True)
    if fault==2:request('ledger','/control',{'pool_size':1,'db_hold':hold},True)
    if fault==3:request('ledger','/control',{'delay_seconds':float(rng.uniform(.26,.34) if shift else rng.uniform(.12,.23)),'db_hold':hold},True)
    return int(rng.integers(7,10) if shift else rng.integers(4,7))

def batch(n):
    def one(_):
        try:request('checkout','/request');return True
        except Exception:return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:ok=list(ex.map(one,range(n)))
    return sum(ok)

def frame(n):
    start=time.monotonic();ok=batch(n);elapsed=time.monotonic()-start
    m={s:request(s,'/metrics') for s in SERVICES}
    c,p,l=(m[s] for s in SERVICES)
    # Fixed physical normalization; no fault-conditioned transforms.
    vector=[c['latency_p95']/3,c['error_rate'],min(1,n/12),
            p['cpu_mean']/.15,l['cpu_mean']/.15,
            l['pool_wait_p95']/2,l['db_mean']/.2,
            p['self_p95']/3,l['latency_p95']/3,p['downstream_p95']/3]
    return {'vector':[min(2,max(0,x)) for x in vector],'metrics':m,'batch_seconds':elapsed,'successes':ok,'requests':n}
