"""Summarize training audit with incidents, rather than candidates, as units."""
import json,itertools
from pathlib import Path
import numpy as np

def summarize(records):
    ids=sorted({r['id'] for r in records});per=[]
    assert len(ids)==60
    for identity in ids:
        rows=[r for r in records if r['id']==identity];assert len(rows)==15
        values={k:float(np.mean([r[k] for r in rows])) for k in ('values_forecast_mse','values_mean_mse','masks_forecast_mse','masks_mean_mse','probability_brier')}
        values['entropy_mae']=float(np.mean([abs(r['predicted_entropy']-r['posterior_entropy']) for r in rows]))
        values['entropy_bias']=float(np.mean([r['predicted_entropy']-r['posterior_entropy'] for r in rows]))
        concordant=[];shared_concordant=[];regret=[];correct=[];bestcorrect=[]
        for budget in range(5):
            candidates=[r for r in rows if r['budget']==budget];assert len(candidates)==5-budget
            assert len({r['action'] for r in candidates})==5-budget
            chosen=min(candidates,key=lambda r:r['predicted_entropy'])
            regret.append(chosen['shared_nll']-min(r['shared_nll'] for r in candidates))
            correct.append(chosen['shared_correct']);bestcorrect.append(any(r['shared_correct'] for r in candidates))
            for a,b in itertools.combinations(candidates,2):
                predicted=np.sign(a['predicted_entropy']-b['predicted_entropy'])
                for key,dest in [('posterior_entropy',concordant),('shared_nll',shared_concordant)]:
                    actual=np.sign(a[key]-b[key]);dest.append(.5 if actual==0 or predicted==0 else float(actual==predicted))
        values.update(entropy_pair_concordance=float(np.mean(concordant)),shared_score_pair_concordance=float(np.mean(shared_concordant)),shared_nll_regret=float(np.mean(regret)),selected_check_accuracy=float(np.mean(correct)),best_check_accuracy=float(np.mean(bestcorrect)))
        per.append(values)
    return dict(cases=len(ids),candidate_records=len(records),**{k:float(np.mean([v[k] for v in per])) for k in per[0]})

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'reports/evidence/rcaeval-forecast-audit-v1'
    result={str(seed):summarize(json.loads((root/f'audit-{seed}/records.json').read_text())) for seed in (42,43)}
    (root/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
