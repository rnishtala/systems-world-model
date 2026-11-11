"""Replay parity then fresh live incidents through the inference-only HTTP API."""
import argparse,json,time,uuid,urllib.error
from pathlib import Path
import numpy as np
from microservice.evaluate import agent
from microservice.environment import Incident
from microservice.latent_probe import observations
from microservice.mechanisms import FAULT_MASK
from microservice.live import configure,frame,reset,wait_ready,batch,reset_measurement_window

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--package',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);ref=Path(a.reference)
    readout=json.loads((Path(a.package)/'readout-23.json').read_text())['readout_sha256']
    rows=list(map(json.loads,(ref/'test.jsonl').read_text().splitlines()));expected=json.loads((ref/'predictions.json').read_text())['latent']
    replay=[]
    for i,row in enumerate(rows):
        incident=str(uuid.uuid4());actions=[]
        for t,v in enumerate(observations(row)):
            r=agent(23,v,t==0,0. if t==0 else -.2,incident);actions.append(r['action'])
        replay.append(dict(episode=i,expected=expected[i],actual=r['action']-4,actions=actions,readout_sha256=r.get('readout_sha256')))
    parity=all(x['actions'][:3]==[1,2,3] and x['actual']==x['expected'] and x['readout_sha256']==readout for x in replay)
    # Test the real HTTP contract, not a Python-only validator.
    import urllib.request
    rejected=[]
    base=dict(seed=23,vector=[0.]*14,first=True,reward=0.,incident_id=str(uuid.uuid4()))
    for body in ({**base,'fault':7},{**base,'reward':3.}):
        try:
            request=urllib.request.Request('http://swm-inference:8088/act',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=30) as response:rejected.append(False)
        except urllib.error.HTTPError as e:rejected.append(e.code==400)
    result=dict(n=len(replay),matching=sum(x['actual']==x['expected'] for x in replay),passed=parity and all(rejected),contract_rejections=rejected,rows=replay)
    (out/'parity.json').write_text(json.dumps(result,indent=2)+'\n');print('PARITY',result['matching'],result['passed'],flush=True)
    if not result['passed']:
        (out/'summary.json').write_text(json.dumps(dict(parity_passed=False,fresh_skipped=True,gate_passed=False),indent=2)+'\n');return
    rng=np.random.default_rng(6901);faults=np.tile(np.arange(8),10);rng.shuffle(faults);collected=[];wait_ready()
    try:
        with (out/'live.jsonl').open('x') as f:
            for i,fault in enumerate(faults):
                start=time.monotonic();n=configure(int(fault),rng,True)
                for _ in range(2):
                    if batch(n)!=n:raise RuntimeError('Warmup request failure')
                reset_measurement_window();frames=[frame(n)];state=Incident(frames,0,compositional=True);v=state.observe().tolist();trace=[];incident=str(uuid.uuid4())
                for t in range(4):
                    r=agent(23,v,t==0,0. if t==0 else -.2,incident);trace.append(dict(observation=v,response=r))
                    if time.monotonic()-start>35:raise RuntimeError('Incident exceeded fault lease safety window')
                    if r['controller']!='inspect-latent' or r['action_contract']!='mechanisms-v1':raise RuntimeError('Wrong serving contract')
                    if t<3:
                        if r['action']!=t+1:raise RuntimeError('Unexpected inspection order')
                        frames.append(frame(n));v=state.step(r['action'])[0].tolist()
                if not 4<=r['action']<12 or r.get('readout_sha256')!=readout:raise RuntimeError('Invalid diagnosis or readout')
                for j,x in enumerate(frames):
                    if x['successes']!=n or any(m['count']!=(j+1)*n for m in x['metrics'].values()):raise RuntimeError('Invalid measurement window')
                target=FAULT_MASK[int(fault)];diagnosis=r['action']-4
                row=dict(episode=i,scenario_seed=6901,fault=int(fault),target=target,diagnosis=diagnosis,correct=diagnosis==target,seconds=time.monotonic()-start,frames=frames,trace=trace)
                collected.append(row);f.write(json.dumps(row)+'\n');f.flush();print('LIVE',i,int(fault),diagnosis,flush=True)
    finally:reset()
    families={str(f):dict(n=10,correct=sum(r['correct'] for r in collected if r['fault']==f)) for f in range(8)}
    result=dict(parity_passed=True,fresh_skipped=False,n=len(collected),correct=sum(r['correct'] for r in collected),families=families,gate_passed=all(v['correct']>=(10 if f=='0' else 9) for f,v in families.items()))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps(result),flush=True)
if __name__=='__main__':main()
