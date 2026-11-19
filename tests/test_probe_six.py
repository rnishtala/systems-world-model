import numpy as np
from pathlib import Path
from microservice.probe_six import load_readouts,score_rows
from microservice.partial_readout import score,diagnose
import json

def test_frozen_models_and_six_prefix_ignore_later_measurements():
    root=Path(__file__).resolve().parents[1]/'reports/evidence'
    config,models=load_readouts(root/'probe-six-v1/frozen')
    rows=[json.loads(l) for l in (root/'probe-stopping-v1/collection/train.jsonl').read_text().splitlines()][:32]
    archive=np.load(root/'probe-stopping-v1/evaluation/prefix-features.npz',allow_pickle=False)
    xs={k:archive[k][:32].copy() for k in archive.files}
    _,a=score_rows(xs,rows,models)
    for k in xs:
        if k.endswith(('_6','_7')):xs[k]+=100
    _,b=score_rows(xs,rows,models)
    for n in ('latent6','history6','summary_history6'):assert a['methods'][n]==b['methods'][n]
    assert config['seed']==9302 and config['n']==32
