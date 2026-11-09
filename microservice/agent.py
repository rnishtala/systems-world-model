"""Inference-only HTTP API. No fault controls, labels, or training data mounted."""
import argparse,http.server,json,os,sys,threading,traceback,time,uuid
import numpy as np
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','/results/work/upstream'))
import elements,jax,jax.numpy as jnp,ninjax as nj
from dreamerv3 import main as upstream
from experiment.particles import ParticleActor

class ShapeEnv:
    def __init__(self):
        self.obs_space=dict(vector=elements.Space(np.float32,(14,)),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.int32,(),0,9))
    def close(self):pass

class Model:
    def __init__(self,root,seed):
        c=elements.Config.load(elements.Path(root+f'/train-{seed}/config.yaml')).update(logdir=f'/tmp/inference-{seed}')
        upstream.make_env=lambda config,index,**kw:upstream.wrap_env(ShapeEnv(),config)
        self.agent=upstream.make_agent(c);cp=elements.Checkpoint(elements.Path(root+f'/final-{seed}'));cp.agent=self.agent;cp.load(keys=['agent'])
        self.policy=ParticleActor(self.agent);self.carry=None;self.calls=0;self.sessions={}
        model=self.agent.model
        @nj.pure
        def forecast(dyn,action):
            _,(feat,_)=model.dyn.imagine(dyn,{'action':jnp.full((16,),action,dtype=jnp.int32)},1,False,single=True)
            _,_,pred=model.dec(model.dec.initial(16),feat,jnp.zeros(16,bool),training=False,single=True)
            return pred['vector'].pred().mean(0)
        @nj.pure
        def rewards(dyn):
            dyn=jax.tree.map(lambda x:jnp.repeat(x,4,axis=0),dyn)
            _,(feat,_)=model.dyn.imagine(dyn,{'action':jnp.tile(jnp.arange(4,8,dtype=jnp.int32),16)},1,False,single=True)
            return model.rew(model.feat2tensor(feat),1).pred().reshape(16,4).mean(0)
        self.forecast=jax.jit(lambda params,dyn,act,key:forecast(params,dyn,act,seed=key,create=False,modify=False)[1])
        self.rewards=jax.jit(lambda params,dyn,key:rewards(params,dyn,seed=key,create=False,modify=False)[1])
    def act(self,vector,first,reward,incident_id):
        uuid.UUID(incident_id)
        self.sessions={k:v for k,v in self.sessions.items() if time.monotonic()-v[1]<300}
        if first:
            if incident_id in self.sessions:raise ValueError('incident already active')
            if len(self.sessions)>=16:raise ValueError('too many active incidents')
            self.carry=self.policy.init_policy(1)
        else:
            if incident_id not in self.sessions:raise ValueError('start a new incident')
            self.carry=self.sessions[incident_id][0]
        obs=dict(vector=np.array([vector],np.float32),reward=np.array([reward],np.float32),is_first=np.array([first]),is_last=np.array([False]),is_terminal=np.array([False]))
        self.carry,acts,_=self.policy.policy(self.carry,obs,mode='eval');action=int(acts['action'][0]);self.calls+=1
        result={'action':action}
        with jax.transfer_guard('allow'):
            dyn=self.agent._stack(self.carry)[1];key=jax.random.PRNGKey(self.calls+900)
            if action in (1,2,3):result['predicted_next_vector']=np.asarray(self.forecast(self.agent.params,dyn,np.int32(action),key)).tolist()
            if 4<=action<=7:result['predicted_diagnosis_rewards']=np.asarray(self.rewards(self.agent.params,dyn,key)).tolist()
        if action<4:self.sessions[incident_id]=(self.carry,time.monotonic())
        else:self.sessions.pop(incident_id,None)
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True);a=p.parse_args()
    models={s:Model(a.root,s) for s in a.seeds};lock=threading.Lock()
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
