"""Score frozen acquisition traces once, with exact coverage and budget checks."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--answers',required=True);a=p.parse_args();root=Path(a.evidence)
    hashes=json.loads((root/'frozen-hashes.json').read_text());predictions={};summaries={}
    for seed in (42,43):
        for filename in ('predictions.json','summary.json'):
            name=f'readout-{seed}/{filename}';assert hashlib.sha256((root/name).read_bytes()).hexdigest()==hashes[name],name
        predictions[seed]=json.loads((root/f'readout-{seed}/predictions.json').read_text());summaries[seed]=json.loads((root/f'readout-{seed}/summary.json').read_text())
        assert not summaries[seed]['smoke_training_case_only'] and not summaries[seed]['test_labels_loaded']
    assert summaries[42]['classifier_sha256']==summaries[43]['classifier_sha256']
    assert [r for r in predictions[42] if r['method']!='dreamer']==[r for r in predictions[43] if r['method']!='dreamer']
    # First access to held-out truth: all prediction files have already been frozen.
    truth={r['id']:r for r in json.loads(Path(a.answers).read_text()) if r['split']=='test'};ids=sorted(truth);n=len(ids)
    output={};gates={};all_full=None
    for seed,rows in predictions.items():
        indexed={}
        for r in rows:
            key=(r['id'],r['method'],r['replicate'],r['budget']);assert key not in indexed;indexed[key]=r
            assert r['id'] in truth and len(r['checks'])==r['budget']==len(set(r['checks']))
            assert len(r['ranks'])==5 and len(set(r['ranks']))==5
        expected={(cid,m,rep,b) for cid in ids for m in ('dreamer','fixed','retrieval','random') for rep in range(16 if m=='random' else 1) for b in range(1,6)}
        assert set(indexed)==expected,'Missing or unexpected incident/method/budget predictions'
        results={};case_correct={}
        for method in ('dreamer','fixed','retrieval','random'):
            reps=16 if method=='random' else 1
            accuracy={};top3={};case_correct[method]={}
            for budget in range(1,6):
                per_case=[];per_case3=[]
                for cid in ids:
                    hits=[];hits3=[]
                    for rep in range(reps):
                        r=indexed[cid,method,rep,budget]
                        if budget>1:assert r['checks'][:-1]==indexed[cid,method,rep,budget-1]['checks']
                        hits.append(r['ranks'][0]==truth[cid]['service']);hits3.append(truth[cid]['service'] in r['ranks'][:3])
                        if budget==5:
                            full=indexed[cid,'fixed',0,5]['ranks'];assert r['ranks']==full,'Full information must yield identical shared diagnosis'
                    per_case.append(float(np.mean(hits)));per_case3.append(float(np.mean(hits3)))
                accuracy[str(budget)]=float(np.mean(per_case));top3[str(budget)]=float(np.mean(per_case3));case_correct[method][str(budget)]=per_case
            full=accuracy['5'];target=full-1/n-1e-9
            results[method]=dict(accuracy_by_budget=accuracy,top3_by_budget=top3,mean_accuracy_b1_4=float(np.mean([accuracy[str(b)] for b in range(1,5)])),minimum_budget_within_one_case_of_full=min(b for b in range(1,6) if accuracy[str(b)]>=target),budget3_within_one_case_of_full=accuracy['3']>=target)
        full_correct=round(results['fixed']['accuracy_by_budget']['5']*n)
        if all_full is None:all_full=full_correct
        assert all_full==full_correct
        gates[str(seed)]={m:results['dreamer']['mean_accuracy_b1_4']-results[m]['mean_accuracy_b1_4']>=.05-1e-9 for m in ('fixed','retrieval')}
        output[str(seed)]=dict(methods=results,full_information_correct=full_correct,case_ids=ids,case_correct=case_correct)
    result=dict(test_cases=n,seeds=output,gate_comparisons=gates,diagnosis_ceiling_passed=all_full>=n-1,primary_gate_passed=all_full>=n-1 and all(v for s in gates.values() for v in s.values()),same_classifier_and_non_dreamer_predictions_across_seeds=True)
    out=root/'scores.json';assert not out.exists(),'Do not overwrite confirmation scores';out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(test_cases=n,primary_gate_passed=result['primary_gate_passed'],full_information_correct=all_full,methods={s:v['methods'] for s,v in output.items()}),indent=2))
if __name__=='__main__':main()
