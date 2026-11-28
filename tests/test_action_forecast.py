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


def test_fresh_comparison_schedule_and_grouped_scores():
    from microservice.action_compare import fresh_cases,paired_report
    from microservice.action_balanced import schedule
    cases=fresh_cases(12345,schedule(112))
    assert len(cases)==40 and len({c['group'] for c in cases})==10
    assert all(c['split']=='test' for c in cases)
    with pytest.raises(ValueError,match='overlap'):fresh_cases(112,schedule(112))
    rows=[dict(c,outcome=[0,0,0,0],predictions={'dreamer':[.1,0,0,0],'history_ridge':[.2,0,0,0]}) for c in cases]
    r=paired_report(rows,'history_ridge')
    assert r['profiles']==10 and r['wins']==10 and r['substantial_improvement']
    assert r['dreamer_minus_baseline']==pytest.approx(-.1)


def test_comparison_queue_requires_owned_restored_checkpoint(tmp_path):
    from launcher.store import Store
    from launcher.common import job_manifest
    from tests.test_launcher import ARGS,IMAGE
    s=Store(tmp_path/'queue');old=s.create('a@example.com',ARGS,IMAGE,'test')
    old.update(status='completed',restored=True,action_dreamer_source='b'*12);s.save(old)
    r=s.create('a@example.com',ARGS,IMAGE,'test',temporal_pilot=True,action_comparison_source=old['id'])
    command=job_manifest(r,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert command[command.index('--action-comparison-source')+1]==old['id']
    with pytest.raises(ValueError):s.create('b@example.com',ARGS,IMAGE,'test',temporal_pilot=True,action_comparison_source=old['id'])
    with pytest.raises(ValueError):s.create('a@example.com',ARGS,IMAGE,'test',temporal_pilot=True,action_comparison_source=old['id'],action_inference_source=old['id'])


def test_comparison_collects_before_offline_inference_without_outcome_input(tmp_path,monkeypatch):
    from microservice import action_compare as m
    from microservice.action_balanced import schedule
    from tests.test_action_outcomes import window
    import sys
    original=tmp_path/'aaaaaaaaaaaa'/'temporal-pilot'
    original.mkdir(parents=True);source=tmp_path/'bbbbbbbbbbbb'/'temporal-pilot';source.mkdir(parents=True)
    out=tmp_path/'cccccccccccc'/'temporal-pilot'
    raw=json.dumps({'models':{'history_ridge':{}},'selected':'history_ridge','cv':{}}).encode()
    (original/'frozen-baselines.json').write_bytes(raw);(original/'examples.json').write_bytes(b'[]')
    (original/'protocol.json').write_text(json.dumps(dict(seed=112,balanced=True,cases=schedule(112))))
    (original/'summary.json').write_text(json.dumps({'action_outcomes':dict(all_restored=True,frozen_sha256=hashlib.sha256(raw).hexdigest())}))
    monkeypatch.setattr(m,'verify_source',lambda _:dict(source_run='aaaaaaaaaaaa',source_examples_sha256=hashlib.sha256(b'[]').hexdigest(),training_sha256='training'))
    monkeypatch.setattr(m,'checkpoint_hash',lambda _:'checkpoint')
    calls=[]
    class Model:
        def __init__(self,source):pass
        def __call__(self,h,action):
            assert np.asarray(h).shape==(4,len(FEATURES))
            assert len(calls) in (0,40)
            return [0,1,.1,0]
    monkeypatch.setattr(m,'Forecaster',Model)
    monkeypatch.setattr(m.a,'predictions',lambda frozen,row:{'history_ridge':[0,1,.1,0]})
    def incident(mechanism,action,profile,before_action):
        i=len(calls);windows=[window(3+j*.5,j+1) for j in range(6)]
        before_action(windows)
        saved=json.loads((out/f'prediction-{i}.json').read_text())
        assert 'outcome' not in saved and 'dreamer' not in saved['predictions']
        assert (out/'protocol.json').exists()
        calls.append(i);windows.extend(window(12+j*.5,20+j*2,0) for j in range(5))
        return dict(windows=windows,restored=True)
    monkeypatch.setattr(m.a,'incident',incident);monkeypatch.setenv('POD_NAMESPACE','systems-world-model-lab')
    monkeypatch.setattr(sys,'argv',['compare','--source',str(source),'--out',str(out),'--seed','12345'])
    m.main();result=json.loads((out/'summary.json').read_text())['action_comparison']
    assert result['test_trials']==40 and result['training_trials']==0
    assert set(result['methods'])=={'history_ridge','dreamer'} and result['checkpoint_unchanged']


def test_sequential_training_and_rollout_align_without_future_observations():
    from microservice.action_model import frame, next_action

    truth = np.array([.8,.5,.2,0.])
    row = dict(observations=np.ones((4,len(FEATURES))),outcome=truth,action='load')
    imagined = []
    seen = []
    def observe(carry,tokens,previous,reset,**kw):
        seen.append(tokens)
        return carry,None,0
    def imagine(carry,controls,length,**kw):
        imagined.append(controls['action'])
        return carry+1,(carry+1,None)
    class Decoder:
        symlog=False
        def __call__(self,carry,feature,reset,**kw):
            raw=np.zeros((PARTICLES,WIDTH))
            if feature:
                raw[:,len(FEATURES)]=frame(row,3+feature,sequential=True)[len(FEATURES)]
            return carry,None,{'vector':types.SimpleNamespace(pred=lambda:raw)}
    inner=types.SimpleNamespace(init_policy=lambda n:(None,0,None,{}),
        enc=lambda c,o,r,**kw:(c,None,o['vector']),dyn=types.SimpleNamespace(observe=observe,imagine=imagine),dec=Decoder())
    values,finite=prior_outcomes(inner,prefix_frames(row['observations']),2*next_action(row,3)-1,np,sequential=True)
    np.testing.assert_allclose(values,truth,atol=1e-6)
    assert finite and len(seen)==4 and len(imagined)==4
    for i,control in enumerate(imagined):
        np.testing.assert_array_equal(control,np.tile(2*next_action(row,3+i)-1,(PARTICLES,1)))
