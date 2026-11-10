"""Forced-inspection representation probe, NOT autonomous or held-out success."""
import argparse,json
from pathlib import Path
import numpy as np
import jax
import experiment.particles  # Load this bundle before upstream main modifies sys.path.
from microservice.agent import Model
from microservice.environment import GROUPS,normalize
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--scaled',action='store_true');a=p.parse_args()
if Path(a.out).exists():raise SystemExit('Fresh output required')
model=Model(a.root,a.seed);rows=[]
for episode,record in enumerate(json.loads(x) for x in Path(a.data).read_text().splitlines()):
    frames=record['frames'];values=[0.]*7;seen=[0.]*3;carry=model.policy.init_policy(1)
    for step in range(4):
        raw=frames[min(step,3)]['vector'];f=normalize(raw) if a.scaled else raw
        obs=dict(vector=np.array([[*f[:3],*values,*seen,step/7]],np.float32),reward=np.array([0 if step==0 else -.2],np.float32),is_first=np.array([step==0]),is_last=np.array([False]),is_terminal=np.array([False]))
        carry,acts,_=model.policy.policy(carry,obs,mode='eval')
        if step<3:
            action=step+1
            nxt=frames[min(step+1,3)]['vector'];nxt=normalize(nxt) if a.scaled else nxt
            for i in GROUPS[action]:values[i-3]=nxt[i]
            seen[action-1]=1
            carry=(*carry[:-1],{**carry[-1],'action':[jax.device_put(np.int32(action),ref.sharding) for ref in carry[-1]['action']]})
    with jax.transfer_guard('allow'):
        scores=np.asarray(model.rewards(model.agent.params,model.agent._stack(carry)[1],jax.random.PRNGKey(episode)))
    rows.append({'fault':record['fault'],'actor_choice':int(acts['action'][0])-4,'world_model_choice':int(scores.argmax()),'rewards':scores.tolist()})
result={'scope':'Forced inspections on calibration incidents; not autonomous/live/held-out accuracy.','actor_accuracy':float(np.mean([r['actor_choice']==r['fault'] for r in rows])),'world_model_reward_ranking_accuracy':float(np.mean([r['world_model_choice']==r['fault'] for r in rows])),'rows':rows}
Path(a.out).write_text(json.dumps(result,indent=2));print('PROBE',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
