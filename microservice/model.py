"""Load a frozen probe-target Dreamer checkpoint for posterior extraction."""
import json
from pathlib import Path
import numpy as np
from microservice.configuration import elements,upstream
from microservice.probe_target import ProbeActor
from microservice.supervision import make_agent

class ShapeEnv:
    def __init__(self):
        self.obs_space=dict(vector=elements.Space(np.float32,(14,)),mechanism_target=elements.Space(np.float32,(3,),0,1),reward=elements.Space(np.float32),is_first=elements.Space(bool),is_last=elements.Space(bool),is_terminal=elements.Space(bool))
        self.act_space=dict(reset=elements.Space(bool),action=elements.Space(np.float32,(2,),0,1))
    def close(self):pass

class Model:
    def __init__(self,root,seed):
        root=Path(root)
        meta=json.loads((root/f'training-{seed}.json').read_text())
        if meta['action_contract']!='probe-target-v1' or not meta['mechanism_supervision']:
            raise ValueError('Expected a supervised probe-target-v1 checkpoint')
        config=elements.Config.load(elements.Path(str(root/f'train-{seed}/config.yaml'))).update(logdir=f'/tmp/inference-{seed}')
        upstream.make_env=lambda config,index,**kw:upstream.wrap_env(ShapeEnv(),config)
        self.agent=make_agent(config)
        checkpoint=elements.Checkpoint(elements.Path(str(root/f'final-{seed}')))
        checkpoint.agent=self.agent;checkpoint.load(keys=['agent'])
        self.policy=ProbeActor(self.agent)
