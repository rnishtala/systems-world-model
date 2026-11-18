"""Inference-only HTTP API. No fault controls, labels, or training data mounted."""
import argparse,http.server,json,os,sys,threading,traceback,time,uuid
import numpy as np
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','/results/work/upstream'))
import elements,jax,jax.numpy as jnp,ninjax as nj
from experiment.particles import ParticleActor
from microservice.guard import assess, GROUPS
from pathlib import Path
from dreamerv3 import main as upstream

class ShapeEnv:
    def __init__(self,count=9,supervised=False,workload=False):
        self.obs_space=dict(vector=elements.Space(np.float32,(14,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        if supervised:self.obs_space['mechanism_target']=elements.Space(np.float32,(3,),0,1)
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.float32,(1,),0,1) if workload else elements.Space(np.int32,(),0,count))
    def close(self):pass

class Model:
    def __init__(self,root,seed,controller="native"):
        self.controller=controller
        meta=json.loads((Path(root)/f'training-{seed}.json').read_text())
        self.supervised=meta.get('mechanism_supervision',False)
        if controller=='inspect-mechanisms' and not self.supervised:raise ValueError('Auxiliary head missing')
        self.contract=meta.get('action_contract','single-v1')
        if self.contract not in ('single-v1','mechanisms-v1','workload-continuous-v1'):raise ValueError('Unknown action contract')
        self.classes=8 if self.contract!='single-v1' else 4
        if self.contract=='workload-continuous-v1' and controller!='native':raise ValueError('Workload checkpoint is offline only')
        if self.classes==8 and controller=='inspect-guarded':raise ValueError('Guard is calibrated only for single-v1')
        self.guard=json.loads((Path(__file__).parent/"calibration/predictive-guard-v1.json").read_text()) if controller=="inspect-guarded" else None
        from microservice.latent_readout import LatentReadout
        if controller=='inspect-latent' and self.contract!='mechanisms-v1':raise ValueError('Latent readout requires mechanisms-v1')
        self.readout=LatentReadout(root,seed) if controller=='inspect-latent' else None
        self.seed=seed
        c=elements.Config.load(elements.Path(root+f'/train-{seed}/config.yaml')).update(logdir=f'/tmp/inference-{seed}')
        upstream.make_env=lambda config,index,**kw:upstream.wrap_env(ShapeEnv(self.classes+5,self.supervised,self.contract=='workload-continuous-v1'),config)
        from microservice.supervision import make_agent
        self.agent=(make_agent if self.supervised else upstream.make_agent)(c);cp=elements.Checkpoint(elements.Path(root+f'/final-{seed}'));cp.agent=self.agent;cp.load(keys=['agent'])
        from microservice.workload import WorkloadActor
        self.policy=WorkloadActor(self.agent) if self.contract=='workload-continuous-v1' else ParticleActor(self.agent);self.carry=None;self.calls=0;self.sessions={}
        model=self.agent.model
        @nj.pure
        def forecast(dyn,action):
            _,(feat,_)=model.dyn.imagine(dyn,{'action':jnp.full((16,1) if self.contract=='workload-continuous-v1' else (16,),action,dtype=jnp.float32 if self.contract=='workload-continuous-v1' else jnp.int32)},1,False,single=True)
            _,_,pred=model.dec(model.dec.initial(16),feat,jnp.zeros(16,bool),training=False,single=True)
            return pred['vector'].pred().mean(0)
        @nj.pure
        def rewards(dyn):
            dyn=jax.tree.map(lambda x:jnp.repeat(x,self.classes,axis=0),dyn)
            _,(feat,_)=model.dyn.imagine(dyn,{'action':jnp.tile(jnp.arange(4,4+self.classes,dtype=jnp.int32),16)},1,False,single=True)
            return model.rew(model.feat2tensor(feat),1).pred().reshape(16,self.classes).mean(0)
        @nj.pure
        def mechanisms(dyn):
            return jax.nn.sigmoid(model.mechanisms(model.feat2tensor(dyn))).mean(0)
        self.mechanisms=jax.jit(lambda params,dyn,key:mechanisms(params,dyn,seed=key,create=False,modify=False)[1])
        self.forecast=jax.jit(lambda params,dyn,act,key:forecast(params,dyn,act,seed=key,create=False,modify=False)[1])
        self.rewards=jax.jit(lambda params,dyn,key:rewards(params,dyn,seed=key,create=False,modify=False)[1])
    def act(self,vector,first,reward,incident_id):
        if self.contract=='workload-continuous-v1':raise ValueError('Offline workload checkpoint cannot serve diagnosis actions')
        uuid.UUID(incident_id)
        self.sessions={k:v for k,v in self.sessions.items() if time.monotonic()-v[1]<300}
        if first:
            if incident_id in self.sessions:raise ValueError('incident already active')
            if len(self.sessions)>=16:raise ValueError('too many active incidents')
            self.carry=self.policy.init_policy(1);history=[];pending=None
        else:
            if incident_id not in self.sessions:raise ValueError('start a new incident')
            self.carry,_,history,pending=self.sessions[incident_id]
        if pending:
            history=history+[{**pending,"actual":[vector[i] for i in pending["indices"]]}]
        obs=dict(vector=np.array([vector],np.float32),reward=np.array([reward],np.float32),is_first=np.array([first]),is_last=np.array([False]),is_terminal=np.array([False]))
        if self.supervised:obs['mechanism_target']=np.zeros((1,3),np.float32)
        self.carry,acts,_=self.policy.policy(self.carry,obs,mode='eval');action=int(acts['action'][0]);self.calls+=1
        if self.controller in ('inspect-first','inspect-guarded','inspect-mechanisms','inspect-latent'):
            # Fixed evidence acquisition, learned diagnosis; no expert thresholds.
            missing=[i+1 for i in range(3) if not vector[10+i]]
            with jax.transfer_guard('allow'):
                dyn=self.agent._stack(self.carry)[1]
                scores=np.asarray(self.rewards(self.agent.params,dyn,jax.random.PRNGKey(self.calls+900)))
            action=missing[0] if missing else int(scores.argmax())+4
            if not missing and self.controller=='inspect-mechanisms':
                with jax.transfer_guard('allow'):
                    probabilities=np.asarray(self.mechanisms(self.agent.params,dyn,jax.random.PRNGKey(self.calls+900)))
                action=4+int(np.sum((probabilities>.5)*np.array([1,2,4])))
            if not missing and self.readout is not None:
                from microservice.latent_readout import state_features
                with jax.transfer_guard('allow'):features=state_features(dyn)
                mask,latent_scores=self.readout.predict(features);action=4+mask
            guard_result=assess(self.guard,self.seed,action-4,history) if not missing and self.guard else None
            if guard_result and guard_result['escalate']:action=8
            previous=[jax.device_put(np.int32(action),ref.sharding) for ref in self.carry[-1]['action']]
            self.carry=(*self.carry[:-1],{**self.carry[-1],'action':previous})
        result={'action':action,'controller':self.controller,'action_contract':self.contract}
        if self.readout is not None and not missing:
            result['latent_scores']=latent_scores.tolist();result['readout_sha256']=self.readout.sha256
        if self.controller=='inspect-mechanisms' and not missing:result['mechanism_probabilities']=probabilities.tolist()
        if self.guard and not missing:
            result['guard']=guard_result;result['predicted_diagnosis_rewards']=scores.tolist()
        with jax.transfer_guard('allow'):
            dyn=self.agent._stack(self.carry)[1];key=jax.random.PRNGKey(self.calls+900)
            if action in (1,2,3):result['predicted_next_vector']=np.asarray(self.forecast(self.agent.params,dyn,np.int32(action),key)).tolist()
            if 4<=action<4+self.classes:result['predicted_diagnosis_rewards']=np.asarray(self.rewards(self.agent.params,dyn,key)).tolist()
        if action in GROUPS:
            indices=list(GROUPS[action]);pending={'indices':indices,'predicted':[result['predicted_next_vector'][i] for i in indices]}
        else:pending=None
        if action<4:self.sessions[incident_id]=(self.carry,time.monotonic(),history,pending)
        else:self.sessions.pop(incident_id,None)
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);p.add_argument('--controller',choices=['native','inspect-first','inspect-guarded','inspect-mechanisms','inspect-latent'],default='native');a=p.parse_args()
    models={s:Model(a.root,s,a.controller) for s in a.seeds};lock=threading.Lock()
    for model in models.values():
        model.act([0.]*14,True,0.,str(uuid.uuid4()))
        with jax.transfer_guard('allow'):
            dyn=model.agent._stack(model.carry)[1];key=jax.random.PRNGKey(123)
            np.asarray(model.forecast(model.agent.params,dyn,np.int32(1),key))
            np.asarray(model.rewards(model.agent.params,dyn,key))
        model.carry=None;model.sessions.clear()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,data,status=200):
            body=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):self.reply({'ready':True,'seeds':list(models)}) if self.path=='/health' else self.reply({},404)
        def do_POST(self):
            if self.path!='/act':return self.reply({},404)
            try:
                n=int(self.headers.get('Content-Length',0))
                if not 0<n<4096:raise ValueError('invalid size')
                d=json.loads(self.rfile.read(n))
                if set(d)!={'seed','vector','first','reward','incident_id'} or len(d['vector'])!=14:raise ValueError('invalid contract')
                if not np.isfinite(d['vector']).all():raise ValueError('nonfinite observation')
                # Only action-cost rewards are accepted; diagnosis correctness is never supplied.
                if float(d['reward']) not in (0.,-.05,-.2):raise ValueError('invalid evaluation reward')
                with lock:r=models[int(d['seed'])].act(d['vector'],bool(d['first']),float(d['reward']),d['incident_id'])
                self.reply(r)
            except Exception:
                traceback.print_exc();self.reply({'error':'invalid request or model failure'},400)
    print('AGENT_READY',a.seeds,flush=True)
    http.server.ThreadingHTTPServer(('0.0.0.0',8088),Handler).serve_forever()
