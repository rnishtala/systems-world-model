import types
import numpy as np
import pytest
from microservice.action_diagnostic import diagnostic,summarize
from microservice.action_model import WIDTH
from microservice.action_forecast import PARTICLES
from microservice.temporal_calibration import FEATURES

def test_outcome_only_enters_reconstruction_branch():
    events=[]
    def enc(state,obs,reset,**kw):
        events.append('encode');return state,None,obs['vector']
    class Dyn:
        free_nats=1.
        def observe(self,state,tokens,prev,reset,**kw):
            events.append('observe');return state,None,{'logit':tokens,'value':tokens}
        def imagine(self,state,action,length,**kw):
            events.append('imagine');v=np.zeros((PARTICLES,WIDTH));return state,({'logit':v,'value':v},action)
        def _dist(self,v):return types.SimpleNamespace(kl=lambda other:np.ones(PARTICLES)*.25)
    class Dec:
        symlog=False
        def __call__(self,state,feat,reset,**kw):return state,None,{'vector':types.SimpleNamespace(pred=lambda:feat['value'])}
    inner=types.SimpleNamespace(init_policy=lambda n:(None,None,None,{}),enc=enc,dyn=Dyn(),dec=Dec())
    frames=np.zeros((5,WIDTH),np.float32);frames[4,len(FEATURES):len(FEATURES)+4]=np.log1p([.3,2,.1,4])
    r=diagnostic(inner,frames,np.zeros(4),np)
    np.testing.assert_allclose(r['prior']['physical_mean'],0)
    np.testing.assert_allclose(r['reconstruction']['physical_mean'],[.3,2,.1,4],rtol=1e-6)
    assert events==['encode','observe']*4+['imagine','encode','observe']
    assert r['kl']==.25 and r['kl_with_free_nats']==1
    s=summarize([r],[[.3,2,.1,4]])
    assert s['reconstruction']['error_fraction']['mae']<1e-6
    assert s['prior']['error_fraction']['mae']==pytest.approx(.3)
    r['prior']['finite']=False
    with pytest.raises(ValueError,match='Nonfinite'):summarize([r],[[.3,2,.1,4]])

def test_diagnostic_queue_isolated_from_other_modes(tmp_path):
    from launcher.store import Store
    from launcher.common import job_manifest
    from tests.test_launcher import ARGS,IMAGE
    s=Store(tmp_path/'queue');old=s.create('a@example.com',ARGS,IMAGE,'test')
    old.update(status='completed',restored=True,action_dreamer_source='b'*12);s.save(old)
    r=s.create('a@example.com',ARGS,IMAGE,'test',temporal_pilot=True,action_diagnostic_source=old['id'])
    command=job_manifest(r,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert command[command.index('--action-diagnostic-source')+1]==old['id']
    with pytest.raises(ValueError):s.create('b@example.com',ARGS,IMAGE,'test',temporal_pilot=True,action_diagnostic_source=old['id'])
    with pytest.raises(ValueError):s.create('a@example.com',ARGS,IMAGE,'test',temporal_pilot=True,action_diagnostic_source=old['id'],action_comparison_source=old['id'])
