"""Verify collection integrity, frozen fits, diagnostic metrics and provenance."""
import hashlib,json
from collections import Counter
import numpy as np
from scripts.evaluate_ledger_probe import O,R,rows,features
from scripts.develop_forecast_readout import choose
from microservice.latent_probe import labels
from microservice.partial_readout import fit,score,diagnose

def main():
    frozen=json.loads((O/'frozen.json').read_text());arrays=np.load(O/'readouts.npz')
    for p,h in frozen['hashes'].items():assert hashlib.sha256((R/p).read_bytes()).hexdigest()==h
    for n,h in json.loads((O/'collector-hashes.json').read_text()).items():assert hashlib.sha256((R/'microservice'/n).read_bytes()).hexdigest()==h
    for split,seed in [('development',9101),('test',9102)]:
        rs=rows(split);assert len(rs)==32
        assert Counter((r['fault'],r['repeat']) for r in rs)==Counter((f,j) for f in range(8) for j in range(4))
        for row in rs:
            assert row['seed']==seed and row['split']==split and 0<row['seconds']<35
            assert (row['utc']<frozen['utc']) if split=='development' else (row['utc']>frozen['utc'])
            assert len(row['base'])==4 and len(row['extra']['ledger'])==len(row['extra']['checkout'])==2
            assert row['order']==(['checkout','ledger'] if row['repeat']<2 else ['ledger','checkout'])
            assert sum(f['requests'] for f in row['extra']['checkout'])==sum(f['requests'] for f in row['extra']['ledger'])==12
            for frame in row['base']+row['extra']['checkout']+row['extra']['ledger']:
                assert frame['requests']==frame['successes'] and frame['requests'] in (3,9)
                for s,m in frame['metrics'].items():
                    assert set(m)=={'count','latency_p95','self_p95','downstream_p95','error_rate'}
                    assert m['count']==(frame['requests'] if frame['target']=='checkout' or s=='ledger' else 0)
                    assert m['error_rate']==0
            # Labels and bookkeeping cannot influence any diagnostic features.
            altered={**row,'fault':99,'seed':0,'repeat':99,'order':['ignored']}
            for n,x in features([row]).items():np.testing.assert_array_equal(x,features([altered])[n])
    dev=rows('development');test=rows('test');dx=features(dev);tx=features(test);y=labels(dev)
    groups=np.array([r['repeat'] for r in dev]);saved=json.loads((O/'predictions.json').read_text())
    target=diagnose(labels(test));np.testing.assert_array_equal(target,saved['target'])
    summary=json.loads((O/'summary.json').read_text());preds={}
    for n,x in dx.items():
        config,counts=choose(x,y,groups);assert config==frozen['selection'][n]['config'] and counts==frozen['selection'][n]['cv_exact_correct']
        m=fit(x,y,config)
        for k,v in m.items():
            if k!='config':np.testing.assert_array_equal(v,arrays[f'{n}_{k}'])
        p=diagnose(score(m,tx[n]));preds[n]=p;np.testing.assert_array_equal(p,saved['methods'][n])
        s=summary['methods'][n];assert s['exact_correct']==int(sum(p==target))
        assert s['pool_delay_correct']==int(sum((p&6)==(target&6)))
        assert s['bit_errors']==[int(sum(((p>>b)&1)!=((target>>b)&1))) for b in range(3)]
        for f in range(8):
            mask=np.array([r['fault']==f for r in test]);fs=s['families'][str(f)]
            assert fs['pool_delay_correct']==int(sum(((p&6)==(target&6))&mask))
            assert fs['exact_correct']==int(sum((p==target)&mask))
    a=summary['methods']['ledger'];b=summary['methods']['checkout']
    gate=bool(a['pool_delay_correct']>=28 and a['pool_delay_correct']>=b['pool_delay_correct']+4 and a['families']['0']['pool_delay_correct']==4 and all(v['pool_delay_correct']>=3 for v in a['families'].values()))
    assert gate==summary['screening_gate']
    (O/'verification.json').write_text(json.dumps(dict(source_hashes=True,balanced_valid_incidents=True,equal_request_budget=True,label_isolation=True,training_selection_reproduced=True,readouts_immutable=True,predictions_metrics_reproduced=True),indent=2)+'\n')
    print('VERIFIED: sources, collection, budgets, label isolation, training-only frozen fits, predictions and metrics.')
if __name__=='__main__':main()
