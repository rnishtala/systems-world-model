"""Bounded mechanism pilot; no model training and no claim of benchmark validity."""
import argparse,concurrent.futures,json,os,time,random
from pathlib import Path
from microservice import live

def drain(timeout=25):
    deadline=time.monotonic()+timeout;quiet=0
    while time.monotonic()<deadline:
        metrics={s:live.request(s,'/metrics') for s in live.SERVICES}
        idle=all(m['counters'].get('active_requests',0)==0 and m['counters'].get('pool_waiters',0)==0 and m['counters'].get('pool_in_use',0)==0 for m in metrics.values())
        quiet=quiet+1 if idle else 0
        if quiet>=3:return metrics
        time.sleep(.2)
    raise RuntimeError('Outstanding requests did not drain; refusing reset')

def configure(mechanism,relief=None):
    # No pool resizing during an incident; controls preserve active work and windows.
    live.request('ledger','/control',dict(pool_size=2,db_hold=.12 if mechanism=='database' and relief!='database' else .04,lease_seconds=45),True)
    live.request('payment','/control',dict(retry_limit=0 if relief=='upstream' else 1,attempt_timeout=.1,request_budget=.25,retry_backoff=.01,lease_seconds=45),True)

def measure(previous):
    result={}
    for service in live.SERVICES:
        old=previous[service];m=live.request(service,'/metrics/window?after='+str(old['sequence']))
        if m['boot_id']!=old['boot_id'] or m['window_gap']:raise RuntimeError('Telemetry continuity lost')
        result[service]=m
    return result

def incident(mechanism,relief):
    if mechanism not in ('healthy','database','upstream') or relief not in ('none','database','upstream'):raise ValueError('Unknown pilot case')
    drain();live.reset();configure('healthy');previous=drain()
    windows=[];pending=set();submitted=0;dropped=0;completed=0;errors=0
    start=time.monotonic();next_request=start;next_window=start+.5;stage=0
    executor=concurrent.futures.ThreadPoolExecutor(max_workers=8)
    try:
        while time.monotonic()-start<12:
            now=time.monotonic();elapsed=now-start
            if stage==0 and elapsed>=3:configure(mechanism);stage=1
            if stage==1 and elapsed>=9:configure(mechanism,relief);stage=2
            rate=120 if mechanism=='upstream' and stage>=1 and not(stage==2 and relief=='upstream') else 2
            if now>=next_request:
                next_request=now+1/rate
                if len(pending)<8:
                    pending.add(executor.submit(live.request,'checkout','/request'));submitted+=1
                else:dropped+=1
            finished={f for f in pending if f.done()}
            for f in finished:
                completed+=1
                try:f.result()
                except Exception:errors+=1
            pending-=finished
            if now>=next_window:
                current=measure(previous)
                windows.append(dict(elapsed=elapsed,metrics=current,client=dict(submitted=submitted,completed=completed,errors=errors,dropped=dropped,inflight=len(pending))))
                previous=current;next_window=now+.5
            time.sleep(.002)
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
        # Client timeouts do not cancel server work. Verify all services are quiet.
        drain();live.reset();restored=drain()
    return dict(mechanism=mechanism,relief=relief,windows=windows,seconds=time.monotonic()-start,restored=all(m['counters'].get('active_requests',0)==0 for m in restored.values()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cases=[(m,r) for m in ('healthy','database','upstream') for r in ('none','database','upstream')];random.Random(a.seed).shuffle(cases)
    protocol=dict(seed=a.seed,cases=cases,scope='Nine-incident mechanism feasibility pilot; no training or accuracy claim',onset_seconds=3,relief_seconds=9,load_seconds=12,max_client_inflight=8,max_retries=1,baseline_requests_per_second=2,burst_requests_per_second=120)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    summaries=[]
    for i,(mechanism,relief) in enumerate(cases):
        result=incident(mechanism,relief)
        (out/f'incident-{i}.json').write_text(json.dumps(result))
        stages={}
        for name,low,high in [('baseline',0,3),('incident',4,9),('relief',10,12)]:
            rows=[w for w in result['windows'] if low<=w['elapsed']<high]
            stages[name]={s:dict(failed_delta=(rows[-1]['metrics'][s]['counters'].get('failed_requests',0)-rows[0]['metrics'][s]['counters'].get('failed_requests',0)) if len(rows)>1 else 0,mean_window_latency=sum(w['metrics'][s]['latency_p95'] for w in rows)/max(1,len(rows)),max_active=max((w['metrics'][s]['counters'].get('active_requests',0) for w in rows),default=0),max_pool_waiters=max((w['metrics'][s]['counters'].get('pool_waiters',0) for w in rows),default=0),attempt_delta=(rows[-1]['metrics'][s]['counters'].get('downstream_attempts',0)-rows[0]['metrics'][s]['counters'].get('downstream_attempts',0)) if len(rows)>1 else 0,timeout_delta=(rows[-1]['metrics'][s]['counters'].get('downstream_timeouts',0)-rows[0]['metrics'][s]['counters'].get('downstream_timeouts',0)) if len(rows)>1 else 0) for s in live.SERVICES}
        summaries.append(dict(mechanism=mechanism,relief=relief,stages=stages,client=result['windows'][-1]['client'],restored=result['restored']))
        print('TEMPORAL_PILOT_PROGRESS',i+1,'/9',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(scope=protocol['scope'],incidents=summaries),indent=2))
    print('TEMPORAL_PILOT_COMPLETE',flush=True)

if __name__=='__main__':main()
