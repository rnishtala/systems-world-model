"""Diagnostic probe, not autonomous success: force checks, then inspect policy/model."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','/results/work/upstream'))
import elements
from dreamerv3 import main as upstream
from experiment.environment import DreamerEnv,QueueTriage

p=argparse.ArgumentParser();p.add_argument('--config',required=True)
p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
p.add_argument('--episodes',type=int,default=90);a=p.parse_args()
c=elements.Config.load(elements.Path(a.config))
upstream.make_env=lambda config,index,**kw:upstream.wrap_env(DreamerEnv(0),config)
agent=upstream.make_agent(c)
cp=elements.Checkpoint(elements.Path(a.checkpoint));cp.agent=agent;cp.load(keys=['agent'])
import jax
import jax.numpy as jnp
import ninjax as nj

@nj.pure
def predict(dyn):
    dyn=jax.tree.map(lambda x:jnp.repeat(x,96,axis=0),dyn)
    actions={'action':jnp.repeat(jnp.arange(3,6,dtype=jnp.int32),32)}
    _,(feat,_)=agent.model.dyn.imagine(dyn,actions,1,False,single=True)
    return agent.model.rew(agent.model.feat2tensor(feat),1).pred().reshape(3,32).mean(1)

predict_jit=jax.jit(lambda params,dyn,seed:predict(params,dyn,seed=seed,create=False,modify=False)[1])
rows=[]
with jax.transfer_guard('allow'):
    for episode in range(a.episodes):
        env=QueueTriage(5000000+episode);obs=env.reset(episode%3)
        carry=agent.init_policy(1);reward=0.
        for t in range(3):
            inp=dict(vector=obs[None],reward=np.array([reward],np.float32),is_first=np.array([t==0]),is_last=np.array([False]),is_terminal=np.array([False]))
            carry,acts,_=agent.policy(carry,inp,mode='eval')
            if t<2:
                action=t+1
                previous={**carry[-1],'action':[jax.device_put(np.int32(action),carry[-1]['action'][0].sharding)]}
                carry=(*carry[:-1],previous)
                obs,reward,_=env.step(action)
        dyn=agent._stack(carry)[1]
        scores=np.asarray(predict_jit(agent.params,dyn,jax.random.PRNGKey(episode)))
        action=int(acts['action'][0]);model_choice=int(scores.argmax())
        rows.append(dict(fault=env.fault,policy_action=action,policy_correct=action==env.fault+3,
                         model_choice=model_choice,model_correct=model_choice==env.fault,
                         predicted_diagnosis_rewards=scores.tolist()))
result={'scope':'Forced two-inspection probe; NOT autonomous diagnostic accuracy.',
        'policy_accuracy_after_checks':float(np.mean([r['policy_correct'] for r in rows])),
        'world_model_reward_ranking_accuracy_after_checks':float(np.mean([r['model_correct'] for r in rows])),
        'rows':rows}
Path(a.out).write_text(json.dumps(result,indent=2))
print('PROBE_RESULT',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
