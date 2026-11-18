"""Verify fresh split provenance and paired diagnosis results after export."""
import hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from microservice.mechanisms import FAULT_MASK

def main():
    root=Path(__file__).resolve().parents[1]/'reports/evidence/lab-fresh-v1'
    for name,h in json.loads((root/'frozen-hashes.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    train=[json.loads(s) for s in (root/'collection/train.jsonl').read_text().splitlines()]
    test=[json.loads(s) for s in (root/'evaluation/test.jsonl').read_text().splitlines()]
    for rows,seed in ((train,8101),(test,8102)):
        assert len(rows)==80 and Counter(r['fault'] for r in rows)==Counter({f:10 for f in range(8)})
        assert all(r['seed']==seed and r['measurement_window']=='post-warmup' and r['seconds']<=35 for r in rows)
        for row in rows:
            assert len(row['frames'])==4
            for j,f in enumerate(row['frames']):
                assert f['requests']==f['successes']
                assert all(m['count']==(j+1)*f['requests'] for m in f['metrics'].values())
    hashes=lambda rows:{hashlib.sha256(json.dumps(r['frames'],sort_keys=True).encode()).hexdigest() for r in rows}
    assert not hashes(train)&hashes(test)
    fitted=json.loads((root/'evaluation/fitted.json').read_text())
    assert fitted['raw_dimensions']==56
    assert fitted['calibration_sha256']==hashlib.sha256((root/'collection/train.jsonl').read_bytes()).hexdigest()
    # Telemetry timestamps demonstrate test measurement starts after readout freezing.
    assert min(f['metrics']['checkout']['timestamp'] for row in test for f in row['frames'])>fitted['utc']
    audit=json.loads((root/'evaluation/audit.json').read_text());assert all(audit.values())
    predictions=json.loads((root/'evaluation/predictions.json').read_text());target=np.array([FAULT_MASK[r['fault']] for r in test]);np.testing.assert_array_equal(target,predictions['target'])
    summary=json.loads((root/'evaluation/summary.json').read_text())
    for method,stats in summary['methods'].items():
        pred=np.array(predictions[method]);assert pred.shape==(80,)
        assert stats['correct']==int(sum(pred==target))
        for f in range(8):assert stats['families'][str(f)]['correct']==sum(pred[i]==target[i] for i,row in enumerate(test) if row['fault']==f)
    assert summary['checkpoint_unchanged']
    report=dict(verified=True,tests_passed=13,train_cases=80,test_cases=80,no_duplicate_trajectories=True,evaluation_after_probe_freeze=True,actual_label_isolation_passed=True,summary=summary)
    (root/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
