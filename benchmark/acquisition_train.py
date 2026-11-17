"""Train action-conditioned Dreamer on random diagnostic inspection trajectories."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from benchmark.acquisition import load_training,ObservationOracle
from benchmark.supervision import TARGET,make_agent,audit


def executed_action(carry,actions):
    import jax
    values=[]
    for i,a in enumerate(actions):
        value=np.asarray(a,np.int32)
        if hasattr(carry[-1]['action'][i],'sharding'):value=jax.device_put(value,carry[-1]['action'][i].sharding)
        values.append(value)
    return (*carry[:-1],{**carry[-1],'action':values})


def main():
    p=argparse.ArgumentParser()
    for name in ('data','train-answers','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=64000);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rows,train,services,columns,g,x,y=load_training(Path(a.data),Path(a.train_answers));dim=2*len(columns)+5
    import sys;sys.path.insert(0,os.environ['DREAMER_SOURCE'])
    import elements,embodied
    from dreamerv3 import main as up
    from experiment.run import configuration
    class Env:
        def __init__(self):self.rng=np.random.default_rng(a.seed);self.done=True
        @property
        def obs_space(self):return dict(vector=elements.Space(np.float32,(dim,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool),**{TARGET:elements.Space(np.float32,(5,),0,1)})
        @property
        def act_space(self):return dict(reset=elements.Space(bool),action=elements.Space(np.int32,(),0,5))
        def step(self,action):
            first=bool(action['reset']) or self.done
            if first:
                i=int(self.rng.integers(len(x)));self.oracle=ObservationOracle(x[i],g);self.target=np.eye(5,dtype=np.float32)[y[i]];self.done=False
                vector=self.oracle.observe()
            else:
                vector=self.oracle.reveal(int(action['action']));self.done=bool(self.oracle.seen.all())
            return dict(vector=vector,reward=np.float32(0),is_first=first,is_last=self.done,is_terminal=self.done,**{TARGET:np.zeros(5,np.float32) if first else self.target})
        def close(self):pass
    c=configuration(a.seed,out/'train',a.steps).update(**{'run.save_every':300})
    up.make_env=lambda config,index,**kw:up.wrap_env(Env(),config)
    captured={};rng=np.random.default_rng(a.seed+9000);state={'order':None,'step':0,'episodes':0}
    def build():
        agent=make_agent(c,5);captured['agent']=agent;policy=agent.policy
        def coverage(carry,obs,mode='train'):
            carry,acts,outs=policy(carry,obs,mode)
            if mode!='train':return carry,acts,outs
            assert len(obs['is_first'])==1
            if obs['is_first'][0]:state.update(order=rng.permutation(5),step=0,episodes=state['episodes']+1)
            actions=acts['action'].copy()
            if not obs['is_last'][0]:
                actions[0]=state['order'][state['step']];state['step']+=1
                carry=executed_action(carry,actions)
            return carry,{**acts,'action':actions},outs
        agent.policy=coverage;return agent
    args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
    Path(c.logdir).mkdir(parents=True);c.save(elements.Path(c.logdir)/'config.yaml');start=time.monotonic()
    embodied.run.train(build,lambda:up.make_replay(c,'replay'),lambda index:up.make_env(c,index),lambda replay,mode:up.make_stream(c,replay,mode),lambda:up.make_logger(c),args)
    agent=captured['agent'];cp=elements.Checkpoint(elements.Path(out/'checkpoint'));cp.agent=agent;cp.save()
    (out/'label-audit.json').write_text(json.dumps(audit(agent,dim,5),indent=2))
    meta=dict(seed=a.seed,steps=a.steps,seconds=time.monotonic()-start,training_cases=len(train),classes=services,columns=columns,dimension=dim,random_coverage_episodes=state['episodes'],scope='Action-conditioned observation acquisition; not remediation; native actor not evaluated',upstream_revision='e01491fad6434b2245a3b8ca201dd7faedcc458c')
    (out/'metadata.json').write_text(json.dumps(meta,indent=2));print('ACQUISITION_TRAINED',json.dumps(meta),flush=True)
if __name__=='__main__':main()
