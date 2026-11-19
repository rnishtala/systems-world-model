"""Experiment operator and telemetry adapter. The policy never gets controls/labels."""
import concurrent.futures,json,time,urllib.request
SERVICES=('checkout','payment','ledger')

def request(service,path,body=None,control=False):
    port=9090 if control else 8080
    req=urllib.request.Request(f'http://swm-{service}.systems-world-model-lab.svc.cluster.local:{port}{path}',data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=12) as r:return json.load(r)

def reset():
    errors=[]
    for s in SERVICES:
        try:request(s,'/control',{},True)
        except Exception as e:errors.append(str(e))
    if errors:raise RuntimeError('Reset failed: '+'; '.join(errors))

def reset_measurement_window():
    """Exclude completed warm-up batches without changing faults or connections.

    Caller must await every warm-up request before opening this window.
    """
    for service in SERVICES:request(service,'/metrics/reset',{},True)

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
    if fault not in range(8):raise ValueError('Unknown experiment fault')
    reset()
    hold=float(rng.uniform(.03,.05))
    request('ledger','/control',{'db_hold':hold},True)
    if fault in (1,4,5,7):request('payment','/control',{'cpu_seconds':float(rng.uniform(.055,.075) if shift else rng.uniform(.025,.05))},True)
    if fault in (2,5,6,7):request('ledger','/control',{'pool_size':1,'db_hold':hold},True)
    if fault in (3,4,6,7):request('ledger','/control',{'delay_seconds':float(rng.uniform(.26,.34) if shift else rng.uniform(.12,.23)),'db_hold':hold,**({'pool_size':1} if fault in (6,7) else {})},True)
    return int(rng.integers(7,10) if shift else rng.integers(4,7))
