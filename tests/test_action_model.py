import copy,json,sys,types
import numpy as np
import pytest
from microservice.action_model import validate,export_training,frame,next_action,ActionEnv,model_spaces,WIDTH
from microservice.action_outcomes import ACTIONS
from microservice.temporal_calibration import FEATURES

def row(split='train',action='database'):
    return dict(split=split,group=split+'-profile',action=action,observations=np.ones((4,len(FEATURES))).tolist(),outcome=[.2,3,.1,2],restored=True)

def test_export_excludes_tests_diagnostics_and_injected_settings(tmp_path):
    source=tmp_path/'source';source.mkdir();rows=[row(s) for s in ('train','test','diagnostic')]
    rows[0]['fault']='must-not-be-exported'
    (source/'examples.json').write_text(json.dumps(rows));(source/'protocol.json').write_text(json.dumps({'cases':rows}))
    (source/'summary.json').write_text(json.dumps({'action_outcomes':{'all_restored':True,'training_trials':1}}))
    meta=export_training(source,tmp_path/'export')
    assert meta['training_trials']==1 and meta['excluded_trials']==2
    data=(tmp_path/'export'/'train.jsonl').read_text()
    assert 'must-not-be-exported' not in data and 'diagnostic' not in data
    with pytest.raises(ValueError):validate([row('test')])
    bad=row();bad['restored']=False
    with pytest.raises(ValueError):validate([bad])
    rows[0]['action']='load';(source/'examples.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='schedule mismatch'):export_training(source,tmp_path/'bad')

def test_prefix_excludes_outcome_and_repair_is_last_transition_only():
    r=row();other=copy.deepcopy(r);other['outcome']=[1,99,9,9]
    for t in range(4):assert np.array_equal(frame(r,t),frame(other,t))
    assert not np.array_equal(frame(r,4),frame(other,4))
    assert frame(r,0).shape==(WIDTH,)
    for t in (0,1,2,4):assert np.argmax(next_action(r,t))==0
    assert np.argmax(next_action(r,3))==ACTIONS.index('database')
    assert np.array_equal(frame(r,4)[:len(FEATURES)],np.zeros(len(FEATURES)))

def test_replay_rejects_unmeasured_actions_and_resets_at_outcome(tmp_path,monkeypatch):
    class Space:
        def __init__(self,*args,**kwargs):pass
    monkeypatch.setitem(sys.modules,'elements',types.SimpleNamespace(Space=Space))
    data=tmp_path/'train.jsonl';data.write_text(json.dumps(row())+'\n')
    env=ActionEnv(data);obs=env.step({'reset':True,'action':np.zeros(4)})
    assert obs['is_first'] and not obs['is_last']
    assert 'scheduled_action' not in model_spaces(env)[0]
    with pytest.raises(ValueError,match='Unmeasured'):env.step({'reset':False,'action':np.eye(4)[3]})
    for t in range(4):
        obs=env.step({'reset':False,'action':next_action(env.row,t)})
        assert obs['is_last']==(t==3)
    obs=env.step({'reset':True,'action':np.zeros(4)})
    assert obs['is_first'] and not obs['is_last']
