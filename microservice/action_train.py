"""Train a fresh upstream Dreamer macro-transition checkpoint on measured training rows."""
import argparse,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--seed',required=True,type=int);p.add_argument('--steps',type=int,default=1000);a=p.parse_args()
    if os.environ.get('POD_NAMESPACE')!='systems-world-model-lab':raise ValueError('Wrong namespace')
    if a.steps!=1000:raise ValueError('Initial smoke experiment is fixed at 1000 steps')
    from microservice.action_model import export_training,ActionEnv,MeasuredActions,make_agent
    from microservice.configuration import configuration,elements,upstream
    import embodied
    out=Path(a.out);meta=export_training(a.source,out);data=out/'train.jsonl'
    (out/'train').mkdir()
    c=configuration(a.seed,out/'train',a.steps)
    upstream.make_env=lambda config,index,**kw:upstream.wrap_env(ActionEnv(data,config.seed*1000+index),config)
    captured={}
    def build():
        agent=make_agent(c,data);policy=MeasuredActions(agent.policy);agent.policy=policy;captured.update(agent=agent,policy=policy);return agent
    args=elements.Config(**c.run,replica=0,replicas=1,logdir=c.logdir,batch_size=c.batch_size,batch_length=c.batch_length,report_length=c.report_length,consec_train=c.consec_train,consec_report=c.consec_report,replay_context=c.replay_context)
    c.save(elements.Path(c.logdir)/'config.yaml');start=time.time()
    embodied.run.train(build,lambda:upstream.make_replay(c,'replay'),lambda index:upstream.make_env(c,index),lambda replay,mode:upstream.make_stream(c,replay,mode),lambda:upstream.make_logger(c),args)
    checkpoint=elements.Checkpoint(elements.Path(out/'final'));checkpoint.agent=captured['agent'];checkpoint.save()
    result=dict(**meta,seed=a.seed,steps=a.steps,seconds=time.time()-start,scheduled_steps=captured['policy'].steps,checkpoint='final',status='training_complete',evaluation='Not evaluated; fresh outcome comparison required',reward='zero; actor outputs overridden by measured actions; no repair policy claim')
    (out/'summary.json').write_text(json.dumps(result,indent=2));print('ACTION_DREAMER_TRAINED',flush=True)

if __name__=='__main__':main()
