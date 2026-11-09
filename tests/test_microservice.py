import numpy as np
from microservice.environment import Incident,GROUPS

def test_no_label_or_unrequested_sensor_exposure():
 frames=[{'vector':[.2,.0,.5,.8,.1,.7,.2,.2,.4,.5]} for _ in range(4)]
 a=Incident(frames,0);b=Incident(frames,3)
 np.testing.assert_array_equal(a.observe(),b.observe())
 assert np.all(a.observe()[3:13]==0)
 for action in (1,2,3):
  av,ar,_=a.step(action);bv,br,_=b.step(action)
  np.testing.assert_array_equal(av,bv);assert ar==br
 assert a.step(4)[1]>0 and b.step(4)[1]<0

def test_controls_cannot_be_selected_as_actions():
 env=Incident([{'vector':[0]*10}]*4,0)
 import pytest
 with pytest.raises(ValueError):env.step(9)
 for _ in range(7):_,_,done=env.step(0)
 assert done and env.choice==-1

def test_manifest_is_namespace_bound_and_keeps_budget():
 import json
 from pathlib import Path
 data=json.loads(Path('deploy/microservice-lab.json').read_text())
 assert all(x['metadata']['namespace']=='systems-world-model-lab' for x in data['items'])
 quota=next(x for x in data['items'] if x['kind']=='ResourceQuota')['spec']['hard']
 assert quota['limits.cpu']=='4' and quota['limits.memory']=='6Gi'
 policies=[x for x in data['items'] if x['kind']=='NetworkPolicy']
 agent=next(x for x in policies if x['metadata']['name']=='swm-agent')
 assert all(p['port']!=9090 for e in agent['spec']['egress'] for p in e.get('ports',[]))
