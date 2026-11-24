import json
import sys
from collections import Counter
import pytest
from microservice import action_outcomes as m
from microservice.action_balanced import schedule,BANDS
from tests.test_action_outcomes import window


def test_balanced_profiles_cover_each_regime_without_split_overlap():
    cases=schedule(123)
    assert cases==schedule(123)
    assert Counter(c['split'] for c in cases)=={'train':64,'test':40,'diagnostic':16}
    profiles={}
    for split in ('train','test'):
        rows=[c for c in cases if c['split']==split]
        profiles[split]={json.dumps(c['profile'],sort_keys=True) for c in rows}
        assert len(profiles[split])==(16 if split=='train' else 10)
        for band,(low,high) in BANDS.items():
            for load in ('low','high'):
                rs=[c for c in rows if c['regime']==band+'-'+load]
                assert len(rs)==(8 if split=='train' else 4)
                assert all(low<=c['profile']['fault_hold']<=high for c in rs)
                assert all((c['profile']['base_rate']*c['profile']['fault_hold']/2<1)==(load=='low') for c in rs)
        for group in {c['group'] for c in rows}:
            assert {c['action'] for c in rows if c['group']==group}==set(m.ACTIONS)
    assert not profiles['train']&profiles['test']
    assert all(json.dumps(c['profile'],sort_keys=True) in profiles['train'] for c in cases if c['split']=='diagnostic')


def test_balanced_fits_training_only_and_scores_test_only(tmp_path,monkeypatch):
    out=tmp_path/'out';fitted=[];calls=[];real_prepare=m.prepare
    def prepare(rows):
        assert len(rows)==64 and all(r['split']=='train' for r in rows)
        fitted.append(len(rows));return real_prepare(rows)
    def incident(mechanism,action,profile,before_action):
        i=len(calls);calls.append(i)
        windows=[window(3+j*.5,j+1) for j in range(6)]
        before_action(windows)
        if i>=64:
            assert (out/f'prediction-{i}.json').exists()
            assert len(fitted)==1
        # Diagnostic repeats fail every request; excluded from held-out scores.
        windows.extend(window(12+j*.5,20+j*2,(20+j*2) if i>=104 else 0) for j in range(5))
        return dict(mechanism=mechanism,relief=action,windows=windows,restored=True)
    monkeypatch.setattr(m,'prepare',prepare);monkeypatch.setattr(m,'incident',incident)
    monkeypatch.setenv('POD_NAMESPACE','systems-world-model-lab')
    monkeypatch.setattr(sys,'argv',['action_outcomes','--out',str(out),'--seed','123','--balanced'])
    m.main()
    summary=json.loads((out/'summary.json').read_text())['action_outcomes']
    assert fitted==[64] and len(calls)==120
    assert summary['training_trials']==64 and summary['test_trials']==40 and summary['independent_test_profiles']==10
    assert summary['balanced']['diagnostic_trials']==16
    assert len(summary['balanced']['repeat_differences'])==8
    assert len(summary['balanced']['paired_profiles'])==10
    assert all(v['mae']['error_fraction']==pytest.approx(0) for v in summary['methods'].values())
    assert len(list(out.glob('prediction-*.json')))==56
