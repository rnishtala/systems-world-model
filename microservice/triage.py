"""Investigate the current lab state without injecting a fault or reading truth."""
import argparse,json,time,uuid
from microservice.evaluate import LiveIncident,agent
from microservice.environment import ACTIONS,DIAGNOSES,GROUPS
p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=10);p.add_argument('--concurrency',type=int,default=6);a=p.parse_args()
if not 1<=a.concurrency<=9:p.error('Diagnostic load is bounded to 1..9 requests per batch')
s=LiveIncident(a.concurrency);incident_id=str(uuid.uuid4());start=time.monotonic();v=s.observe();reward=0.;trace=[];diagnosis=-1
for t in range(7):
    r=agent(a.seed,v,t==0,reward,incident_id);action=r['action'];trace.append({'observation':v,'action':ACTIONS[action],**{k:x for k,x in r.items() if k!='action'}})
    if 4<=action<=7:diagnosis=action-4;break
    if action==8:break
    s.t+=1;v=s.observe(action);reward=-.2 if action in GROUPS else -.05
print(json.dumps({'incident_id':incident_id,'diagnosis':DIAGNOSES[diagnosis] if diagnosis>=0 else 'escalation','checks':s.checks,'seconds':time.monotonic()-start,'trace':trace,'evidence':s.receipts,'scope':'Read-only investigation with bounded synthetic request traffic; no fault injection or ground-truth access.'},indent=2))
