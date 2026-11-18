import json,sys
from collections import Counter
from unittest.mock import patch
import pytest
from microservice import fresh_collect

def test_collection_balanced_clean_and_reset(tmp_path):
    count=[0]
    def reset_window():count[0]=0
    def frame(n):
        count[0]+=n
        return dict(successes=n,requests=n,metrics={s:{'count':count[0]} for s in ('checkout','payment','ledger')},vector=[0.]*10)
    out=tmp_path/'train'
    with patch.object(sys,'argv',['collect','--out',str(out)]),patch.object(fresh_collect,'wait_ready'),patch.object(fresh_collect,'configure',return_value=4),patch.object(fresh_collect,'batch',return_value=4),patch.object(fresh_collect,'reset_measurement_window',side_effect=reset_window),patch.object(fresh_collect,'frame',side_effect=frame),patch.object(fresh_collect,'reset') as reset:
        fresh_collect.main();reset.assert_called_once()
    rows=[json.loads(l) for l in (out/'train.jsonl').read_text().splitlines()]
    assert Counter((r['fault'],r['shifted']) for r in rows)==Counter({(f,s):5 for f in range(8) for s in (False,True)})
    assert all([frame['metrics']['checkout']['count'] for frame in row['frames']]==[4,8,12,16] for row in rows)

def test_failed_case_retained_and_faults_reset(tmp_path):
    out=tmp_path/'failed'
    bad=dict(successes=0,requests=4,metrics={},vector=[0.]*10)
    with patch.object(sys,'argv',['collect','--out',str(out)]),patch.object(fresh_collect,'wait_ready'),patch.object(fresh_collect,'configure',return_value=4),patch.object(fresh_collect,'batch',return_value=4),patch.object(fresh_collect,'reset_measurement_window'),patch.object(fresh_collect,'frame',return_value=bad),patch.object(fresh_collect,'reset') as reset:
        with pytest.raises(RuntimeError,match='Invalid incident'):fresh_collect.main()
        reset.assert_called_once()
    assert len((out/'train.jsonl').read_text().splitlines())==1
    assert not (out/'identity.json').exists()
