"""Independently reproduce frozen probes and all test decisions from exported evidence."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from microservice.temporal import vector
from microservice.partial_readout import fit,score,diagnose
from microservice.temporal_evaluate import select
from microservice.latent_probe import labels
from microservice.mechanisms import FAULT_MASK

R=Path(__file__).resolve().parents[1]/'reports/evidence/lab-temporal-v1'
def wilson(k,n):
    z=1.96;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d
    w=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-w,c+w]

def main():
    frozen=json.loads((R/'evaluation/frozen.json').read_text());params=np.load(R/'evaluation/readouts.npz')
    summary=json.loads((R/'evaluation/summary.json').read_text());saved=json.loads((R/'evaluation/predictions.json').read_text())
    paths={'train':R/'collection/train.jsonl','test':R/'evaluation/test.jsonl'}
    sets={k:[json.loads(l) for l in p.read_text().splitlines()] for k,p in paths.items()}
    assert hashlib.sha256(paths['train'].read_bytes()).hexdigest()==frozen['data_sha256']
    assert hashlib.sha256((R/'evaluation/readouts.npz').read_bytes()).hexdigest()==frozen['readouts_sha256']
    for key,rows in sets.items():
        assert len(rows)==80
        assert all(sum(r['fault']==f for r in rows)==10 for f in range(8))
        assert all(sum(r['fault']==f and r['phase']==p for r in rows)==5 for f in range(8) for p in (0,1))
        for row in rows:
            assert row['seed']==(8601 if key=='train' else 8602)
            assert row['seconds']<=35
            assert row['loads']==(([4,6,4,6] if row['phase']==0 else [6,4,6,4]) if key=='train' else ([7,9,7,9] if row['phase']==0 else [9,7,9,7]))
            for t,f in enumerate(row['frames']):
                assert f['successes']==f['requests']==row['loads'][t]
                assert all(m['count']==f['requests'] for m in f['metrics'].values())
                if key=='test':assert all(m['timestamp']>frozen['utc'] for m in f['metrics'].values())
        fs=np.load(R/f'evaluation/{key}-features.npz')
        expected=np.array([np.array([vector(r,t) for t in range(4)]).reshape(-1) for r in rows])
        np.testing.assert_array_equal(fs['history'],expected)
        np.testing.assert_array_equal(fs['latest'],expected[:,-14:])
        np.testing.assert_array_equal(fs['labels'],labels(rows))
    train=np.load(R/'evaluation/train-features.npz');test=np.load(R/'evaluation/test-features.npz')
    y=labels(sets['train']);families=np.array([r['fault'] for r in sets['train']])
    target=np.array([FAULT_MASK[r['fault']] for r in sets['test']]);assert target.tolist()==saved['target']
    for name in ('history','latest','latent','shuffled_latent'):
        key='latent' if name=='shuffled_latent' else name
        if name=='shuffled_latent':model=fit(train[key],y[np.random.default_rng(8605).permutation(80)],frozen['configs'][name])
        else:
            model,selected=select(train[key],y,families)
            assert selected==frozen['selection'][name]
        for k in ('mean','scale','bias','z','dual'):np.testing.assert_allclose(model[k],params[f'{name}_{k}'],rtol=1e-7,atol=1e-8)
        pred=diagnose(score(model,test[key]));assert pred.tolist()==saved[name]
        assert int(sum(pred==target))==summary['methods'][name]['correct']
    for name,s in summary['methods'].items():
        pred=np.array(saved[name]);assert int(sum(pred==target))==s['correct']
        for f,stats in s['families'].items():
            indices=np.array([r['fault']==int(f) for r in sets['test']])
            assert int(sum((pred==target)&indices))==stats['correct']
    assert all(json.loads((R/'evaluation/audit.json').read_text()).values())
    assert json.loads((R/'restored.json').read_text())['template_unchanged']
    clipping={}
    from microservice.environment import GAIN
    for key,rows in sets.items():
        raw=np.array([f['vector'] for row in rows for f in row['frames']])*GAIN
        clipping[key]={str(i):int(sum(raw[:,i]>2)) for i in (0,1,2,7,8,9)}
    result=dict(clipping_above_upper_bound=clipping,verified=True,scope='Independent readout reproduction, telemetry isolation, collection boundaries, recorded label audit and restoration; latent features not regenerated locally',accuracy_intervals_95={k:wilson(v['correct'],80) for k,v in summary['methods'].items()})
    (R/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
