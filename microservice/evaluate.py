"""Held-out LIVE triage: controls and truth stay in the experiment runner."""
import argparse,json,time,urllib.request,uuid
from pathlib import Path
import numpy as np
from microservice.live import configure,reset,wait_ready,batch,request
from microservice.environment import ACTIONS,GROUPS,DIAGNOSES,normalize

def agent(seed,vector,first,reward,incident_id):
    body=json.dumps(dict(seed=seed,vector=vector,first=first,reward=reward,incident_id=incident_id)).encode()
    with urllib.request.urlopen(urllib.request.Request('http://swm-inference:8088/act',data=body,headers={'Content-Type':'application/json'}),timeout=90) as r:return json.load(r)

class LiveIncident:
    def __init__(self,n):self.n=n;self.t=0;self.checks=0;self.values=[0.]*7;self.seen=[0.]*3;self.receipts=[]
    def observe(self,action=0):

        if batch(self.n)!=self.n:raise RuntimeError('Unplanned request failure; incident is not a valid known-fault trial')
        c=request('checkout','/metrics');v=[c['latency_p95']/3,c['error_rate'],min(1,self.n/12)]
        evidence={'checkout':c}
        if action==1:
            p=request('payment','/metrics');l=request('ledger','/metrics');self.values[:2]=[p['cpu_mean']/.15,l['cpu_mean']/.15];evidence.update(payment=p,ledger=l)
        if action==2:
            l=request('ledger','/metrics');self.values[2:4]=[l['pool_wait_p95']/2,l['db_mean']/.2];evidence['ledger']=l
        if action==3:
            p=request('payment','/metrics');l=request('ledger','/metrics');self.values[4:]=[p['self_p95']/3,l['latency_p95']/3,p['downstream_p95']/3];evidence.update(payment=p,ledger=l)
        if action in GROUPS:self.checks+=1;self.seen[action-1]=1
        self.receipts.append(evidence)
        return normalize(v+self.values)+self.seen+[self.t/7]

def checklist(v):
    if not v[10]:return 1
    if v[3]>.24:return 5
    if not v[11]:return 2
    if v[5]>.2:return 6
    if not v[12]:return 3
    return 7 if v[8]>.35 else 4

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[0]);p.add_argument('--episodes',type=int,default=6);p.add_argument('--scenario-seed',type=int,default=2200);p.add_argument('--shift',action='store_true');p.add_argument('--unknown',action='store_true',help='Separate combined CPU+delay stress test; excluded from known-class acceptance');p.add_argument('--unknown-kind',choices=['cpu-delay','cpu-pool','pool-delay'],default='cpu-delay');p.add_argument('--compositional',action='store_true');a=p.parse_args()
    if a.compositional and a.unknown:raise ValueError('Use compositional all-eight-family evaluation')
    path=Path(a.out)
    if path.exists():raise SystemExit('Refusing to replace evidence')
    path.parent.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(a.scenario_seed);faults=np.full(a.episodes,{'cpu-delay':4,'cpu-pool':5,'pool-delay':6}[a.unknown_kind]) if a.unknown else np.tile(np.arange(8 if a.compositional else 4),a.episodes);rng.shuffle(faults)
    wait_ready()
    try:
        with path.open('x') as f:
            for episode,fault in enumerate(faults):
                n=configure(int(fault),rng,a.shift)
                # Warmup is identical for every fault. Real requests, not synthetic observations.
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup request failed')
                methods=list(a.seeds)+([] if a.compositional else ['checklist']);rng.shuffle(methods)
                incident_started=time.monotonic()
                for method in methods:
                    start=time.monotonic();state=LiveIncident(n);incident_id=str(uuid.uuid4());v=state.observe();first=True;reward=0.;trace=[];diagnosis=-1;predictions=[];ranking=None
                    for t in range(7):
                        if time.monotonic()-incident_started>35:raise RuntimeError('Incident exceeds safe fault lease window')
                        r={'action':checklist(v)} if method=='checklist' else agent(method,v,first,reward,incident_id)
                        if time.monotonic()-incident_started>35:raise RuntimeError('Inference exceeded the safe fault lease window')
                        action=r['action']
                        if method!='checklist' and r.get('action_contract','single-v1')!=('mechanisms-v1' if a.compositional else 'single-v1'):raise ValueError('Incompatible model contract')
                        end=12 if a.compositional else 8
                        entry={'observation':v,'action':str(action) if a.compositional else ACTIONS[action],'controller':r.get('controller','checklist'),'guard':r.get('guard')};trace.append(entry)
                        if 4<=action<end:diagnosis=action-4;ranking=r.get('predicted_diagnosis_rewards');break
                        if action==end:ranking=r.get('predicted_diagnosis_rewards');break
                        state.t+=1;nextv=state.observe(action)
                        if 'predicted_next_vector' in r:
                            indices=list(GROUPS[action]);predictions.append({'action':action,'indices':indices,'predicted':[r['predicted_next_vector'][i] for i in indices],'actual':[nextv[i] for i in indices]})
                        v=nextv;reward=-.2 if action in GROUPS else -.05;first=False
                    from microservice.mechanisms import FAULT_MASK,diagnosis_name
                    target=FAULT_MASK[int(fault)] if a.compositional else int(fault)
                    result=dict(action_contract='mechanisms-v1' if a.compositional else 'single-v1',target=target,observation_version='physical-v2',unknown_fault=a.unknown,episode=episode,method=method,fault=int(fault),diagnosis=diagnosis,diagnosis_name=diagnosis_name(diagnosis) if a.compositional else (DIAGNOSES[diagnosis] if diagnosis>=0 else 'escalation'),correct=bool(diagnosis==target),checks=state.checks,seconds=time.monotonic()-start,since_shared_injection_seconds=time.monotonic()-incident_started,shifted=a.shift,scenario_seed=a.scenario_seed,trace=trace,evidence=state.receipts,predictions=predictions,predicted_diagnosis_rewards=ranking)
                    f.write(json.dumps(result)+'\n');f.flush()
                    print('LIVE',episode,method,int(fault),diagnosis,state.checks,round(result['seconds'],2),flush=True)
    finally:reset()

if __name__=='__main__':main()
