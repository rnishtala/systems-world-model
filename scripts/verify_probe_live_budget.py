"""Reproduce actual-budget counts, frozen predictions, gates and timings."""
import hashlib,json
from pathlib import Path
import numpy as np
from microservice.probe_live_budget import assess,schedule,RNG
from microservice.probe_six import load_readouts
from microservice.budget_features import budget_vector
from microservice.partial_readout import score,diagnose
R=Path(__file__).resolve().parents[1];O=R/'reports/evidence/probe-live-budget-v1'

def main():
    for file,sha in json.loads((O/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((O/file).read_bytes()).hexdigest()==sha,file
    for file,sha in json.loads((O/'source-hashes.json').read_text()).items():assert hashlib.sha256((R/file).read_bytes()).hexdigest()==sha,file
    for file in ('config.json','readouts.npz'):assert (O/'frozen'/file).read_bytes()==(R/'reports/evidence/probe-six-v1/frozen'/file).read_bytes()
    cfg,models=load_readouts(O/'frozen');rows=[json.loads(l) for l in (O/'evaluation/incidents.jsonl').read_text().splitlines()]
    assert len(rows)==128
    cases=[(g,f,r) for g in (0,1) for f in range(8) for r in range(4)];RNG(9401).r.shuffle(cases)
    for i,row in enumerate(rows):
        pair=i//2;g,f,rep=cases[pair]
        assert (row['pair'],row['regime'],row['fault'],row['repeat'],row['scenario_seed'])==(pair,g,f,rep,940100+pair)
        assert row['budget']==([6,8] if pair%2==0 else [8,6])[i%2]
        assert row['schedule']==schedule(g,rep)
        assert len(row['frames'])==row['budget']
        assert row['warmup_inclusive_seconds']<=35 and row['utc']>cfg['utc']
        for k in ('warmup_inclusive_seconds','acquisition_seconds','inference_seconds'):assert np.isfinite(row[k]) and row[k]>0
        for t,frame in enumerate(row['frames']):
            assert [frame['target'],frame['requests']]==row['schedule'][t]
            assert frame['requests']==frame['successes']
            for service,m in frame['metrics'].items():
                expected=frame['requests'] if frame['target']=='checkout' or service=='ledger' else 0
                assert m['count']==expected and m['error_rate']==0
        assert row['measured_requests']==sum(f['requests'] for f in row['frames'])
        assert row['total_requests']==36+row['measured_requests']
    fs=np.load(O/'evaluation/features.npz',allow_pickle=False)
    for budget in (6,8):
        rs=[r for r in rows if r['budget']==budget];hist=[];summaries=[]
        for row in rs:
            seq=np.array([budget_vector(row,t) for t in range(budget)]);summary=[]
            for target in (0,1):
                for load in (.25,.75):
                    seen=seq[(seq[:,4]==target)&np.isclose(seq[:,5],load)]
                    summary.extend(seen[:,:4].mean(0) if len(seen) else np.zeros(4));summary.extend([len(seen),bool(len(seen))])
            hist.append(seq.reshape(-1));summaries.append(np.r_[summary,seq.reshape(-1)])
        np.testing.assert_array_equal(fs[f'history{budget}'],hist);np.testing.assert_array_equal(fs[f'summary_history{budget}'],summaries)
        for n in ('latent','history','summary_history'):
            pred=diagnose(score(models[f'{n}{budget}'],fs[f'{n}{budget}']))
            np.testing.assert_array_equal(pred,[r['predictions'][n] for r in rs])
    result=assess(rows);saved=json.loads((O/'evaluation/summary.json').read_text())
    for k,v in result.items():assert saved[k]==v,k
    assert saved['checkpoint_unchanged'] and saved['readouts_unchanged']
    assert json.loads((O/'checkpoint-hashes.json').read_text())==json.loads((R/'reports/evidence/probe-target-model-v1/checkpoint-hashes.json').read_text())
    (O/'verification.json').write_text(json.dumps(dict(source_and_export_hashes_verified=True,prior_readouts_byte_identical=True,no_unmeasured_frames=True,balanced_paired_scenarios_verified=True,requests_and_timings_verified=True,telemetry_features_reproduced=True,predictions_and_gates_reproduced=True,checkpoint_matches_previous=True),indent=2)+'\n')
    print('VERIFIED')
if __name__=='__main__':main()
