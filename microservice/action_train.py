"""Train a fresh upstream Dreamer macro-transition checkpoint on measured training rows."""
import argparse,json,os,time,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    if a.steps not in (1000,25000):raise ValueError('Use the 1000-step smoke or fixed 25000-step training diagnostic')
    from microservice.action_model import export_training
    out=Path(a.out)
    meta=export_training(a.source,out)
    result=train_partition(out,meta,a.seed,a.steps)
    if a.steps==25000:
        subprocess.run([sys.executable,'-m','microservice.action_diagnostic','--source',str(out),'--out',str(out/'diagnostic')],check=True)
        result['diagnostic']=json.loads((out/'diagnostic'/'summary.json').read_text())['action_diagnostic']
        (out/'summary.json').write_text(json.dumps(result,indent=2))
    print('ACTION_DREAMER_TRAINED',flush=True)

def train_partition(out,meta,seed,steps):
    from microservice.action_model import ActionEnv,MeasuredActions,make_agent,measured_stream
    from microservice.configuration import configuration,elements,upstream
    import embodied
    data=out/'train.jsonl'
    (out/'train').mkdir()
    c=configuration(seed,out/'train',steps)
    if steps==25000:c=c.update({'run.log_every':5})
    upstream.make_env=lambda config,index,**kw:upstream.wrap_env(ActionEnv(data,config.seed*1000+index),config)
    captured={}
    def build():
        agent=make_agent(c,data);original_stream=agent.stream;agent.stream=lambda stream:original_stream(measured_stream(stream));policy=MeasuredActions(agent.policy);agent.policy=policy;captured.update(agent=agent,policy=policy);return agent
    args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
    c.save(elements.Path(c.logdir)/'config.yaml');start=time.time()
    embodied.run.train(build,lambda:upstream.make_replay(c,'replay'),lambda index:upstream.make_env(c,index),lambda replay,mode:upstream.make_stream(c,replay,mode),lambda:upstream.make_logger(c),args)
    checkpoint=elements.Checkpoint(elements.Path(out/'final'));checkpoint.agent=captured['agent'];checkpoint.save()
    result=dict(**meta,seed=seed,steps=steps,seconds=time.time()-start,scheduled_steps=captured['policy'].steps,checkpoint='final',status='training_complete',evaluation='Not evaluated; fresh outcome comparison required',reward='zero; actor outputs overridden by measured actions; no repair policy claim')
    (out/'summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':main()
