import copy
import numpy as np
from microservice.regime import vector,LogIncident,PROFILES,cases,loads
from microservice.regime_evaluate import eligible,select

def row(seconds=.5):
    m={'checkout':{'latency_p95':seconds,'error_rate':0},'payment':{'self_p95':seconds/2,'downstream_p95':seconds/3},'ledger':{'latency_p95':seconds/4}}
    return {'fault':7,'regime':0,'frames':[{'metrics':copy.deepcopy(m),'requests':8} for _ in range(4)]}

def test_no_latency_cap_and_finite_scale():
    values=[vector(row(s),0)[0] for s in (.4,.6,1.2,2.5,8)]
    assert np.all(np.diff(values)>0)
    np.testing.assert_allclose(values,np.log1p(np.array([.4,.6,1.2,2.5,8])/.1),rtol=1e-6)

def test_no_hidden_counters_labels_or_future():
    a=row();b=copy.deepcopy(a);b.update(fault=0,regime=99,seed=999)
    for f in b['frames']:
        f['vector']=[999]*10
        for m in f['metrics'].values():m.update(cpu_mean=999,pool_wait_p95=999,db_mean=999)
    for t in range(4):np.testing.assert_array_equal(vector(a,t),vector(b,t))
    b['frames'][3]['metrics']['checkout']['latency_p95']=10
    for t in range(3):np.testing.assert_array_equal(vector(a,t),vector(b,t))
    assert vector(a,3)[0]!=vector(b,3)[0]

def test_regime_split_and_balancing():
    assert [len(cases(s)) for s in PROFILES]==[128,64,80]
    assert set(sum((list(p) for p in PROFILES['train']),[])).isdisjoint(set(sum((list(p) for p in PROFILES['validation']),[])))
    for split in PROFILES:
        for f in range(8):
            for r in range(len(PROFILES[split])):
                assert sum(x==(f,r,0) for x in cases(split))==sum(x==(f,r,1) for x in cases(split))
                assert loads(split,r,1)==loads(split,r,0)[::-1]

def test_wait_sequence_matches_log_view():
    r=row();sim=LogIncident(r['frames'],7,compositional=True)
    np.testing.assert_array_equal(sim.observe(),vector(r,0))
    for t in range(1,4):np.testing.assert_array_equal(sim.step(0)[0],vector(r,t))

def test_validation_gate_requires_quality_and_comparison():
    d={'n':64,'correct':58,'families':{str(f):{'correct':8 if f==0 else 7} for f in range(8)}}
    results={'latent':d,'history':{'correct':60},'latest':{'correct':50}}
    assert eligible(results)
    for field,value in [('correct',57),('n',80)]:
        bad=copy.deepcopy(results);bad['latent'][field]=value;assert not eligible(bad)
    for f in ('0','1'):
        bad=copy.deepcopy(results);bad['latent']['families'][f]['correct']=5;assert not eligible(bad)
    bad=copy.deepcopy(results);bad['history']['correct']=61;assert not eligible(bad)

def test_selection_does_not_refit_or_use_validation_for_normalization(monkeypatch):
    import microservice.regime_evaluate as m
    real=m.fit;calls=[]
    def tracked(x,y,c):
        assert len(x)==8
        model=real(x,y,c);calls.append(model);return model
    monkeypatch.setattr(m,'fit',tracked)
    x=np.arange(16).reshape(8,2);y=np.tile([1,-1,1],(8,1))
    chosen,_=select(x,y,np.ones((4,2))*999,np.full(4,5))
    assert len(calls)==8 and any(chosen is c for c in calls)
    np.testing.assert_array_equal(chosen['mean'],x.mean(0))
