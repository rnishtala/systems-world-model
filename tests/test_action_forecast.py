import hashlib,json,types
import numpy as np
import pytest
from microservice.action_forecast import prefix_frames,physical_outcomes,prior_outcomes,verify_source,checkpoint_hash,PARTICLES
from microservice.action_model import WIDTH,VERSION
from microservice.temporal_calibration import FEATURES
from microservice.action_outcomes import ACTIONS,TARGETS


def test_prefix_has_no_outcome_or_scheduling_input():
    h=np.ones((4,len(FEATURES)))
    v=prefix_frames(h)
    assert v.shape==(4,WIDTH)
    assert (v[:,len(FEATURES):len(FEATURES)+4]==0).all()
    assert (v[:,-2]==1).all() and (v[:,-1]==0).all()
    for bad in [h[:3],h*np.nan,-h]:
        with pytest.raises(ValueError):prefix_frames(bad)


def test_decoder_inverts_both_transforms():
    y=np.array([.4,12.,.08,3.])
    logged=np.log1p(y);symlog=np.sign(logged)*np.log1p(abs(logged))
    np.testing.assert_allclose(physical_outcomes(symlog),y)
    np.testing.assert_allclose(physical_outcomes(logged,False),y)


def test_prior_observes_four_frames_then_imagines_action_without_future_tokens():
    calls=[];action=2*np.eye(4,dtype=np.float32)[2]-1;h=prefix_frames(np.ones((4,len(FEATURES))))
    def enc(state,obs,reset,**kw):
        calls.append(('encode',obs['vector'].copy()));return state,None,obs['vector']
    def observe(state,tokens,prev,reset,**kw):
        calls.append(('observe',prev['action'].copy()));return state,None,'posterior'
    def imagine(state,acts,length,**kw):
        calls.append(('imagine',acts['action'].copy()));assert length==1;return state,('prior',None)
    values=np.array([.2,4.,.1,2.]);raw=np.zeros((PARTICLES,WIDTH));raw[:,len(FEATURES):len(FEATURES)+4]=np.log1p(np.log1p(values))
    class Decoder:
        symlog=True
        def __call__(self,state,feat,reset,**kw):
            calls.append(('decode',feat));return state,None,{'vector':types.SimpleNamespace(pred=lambda:raw)}
    inner=types.SimpleNamespace(init_policy=lambda n:(None,None,None,{'action':np.zeros((n,4))}),enc=enc,dyn=types.SimpleNamespace(observe=observe,imagine=imagine),dec=Decoder())
    result,finite=prior_outcomes(inner,h,action,np)
    np.testing.assert_allclose(result,values);assert finite
    assert sum(k=='encode' for k,v in calls)==4
    assert sum(k=='observe' for k,v in calls)==4
    assert calls[-1]==('decode','prior')
    np.testing.assert_array_equal(calls[-2][1],np.repeat(action[None],PARTICLES,axis=0))
    observes=[v for k,v in calls if k=='observe']
    for prev in observes[1:]:np.testing.assert_array_equal(prev,np.repeat((2*np.eye(4)[0]-1)[None],PARTICLES,axis=0))


def test_source_identity_and_checkpoint_change_detection(tmp_path):
    data=b'training data';(tmp_path/'train.jsonl').write_bytes(data)
    digest=hashlib.sha256(data).hexdigest()
    meta=dict(version=VERSION,features=FEATURES,targets=list(TARGETS),actions=list(ACTIONS),training_sha256=digest)
    (tmp_path/'manifest.json').write_text(json.dumps(meta));(tmp_path/'summary.json').write_text(json.dumps({'action_dreamer':dict(meta,status='training_complete')}))
    assert verify_source(tmp_path)==meta
    with pytest.raises(ValueError,match='Missing'):checkpoint_hash(tmp_path)
    (tmp_path/'final').mkdir();(tmp_path/'final'/'checkpoint').write_bytes(b'a');first=checkpoint_hash(tmp_path)
    (tmp_path/'final'/'checkpoint').write_bytes(b'b');assert checkpoint_hash(tmp_path)!=first
    (tmp_path/'train.jsonl').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity'):verify_source(tmp_path)
