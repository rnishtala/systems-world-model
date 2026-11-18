import json,hashlib
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'reports/evidence/rcaeval-alignment-v1'
    for name,h in json.loads((root/'frozen-hashes.json').read_text()).items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h
    prior=json.loads((ROOT/'reports/evidence/rcaeval-acquisition-v1/checkpoint-hashes.json').read_text())
    result={}
    for seed in (42,43):
        path=root/f'audit-{seed}';s=json.loads((path/'summary.json').read_text());rows=json.loads((path/'records.json').read_text());features=json.loads((path/'features.json').read_text())
        assert s['n']==60 and len(rows)==1200
        assert not(set(s['fold_ids']['0']) & set(s['fold_ids']['1']))
        assert len(s['fold_ids']['0'])==len(s['fold_ids']['1'])==30
        for name,h in s['checkpoint_hashes'].items():assert prior[f'model-{seed}/'+name]==h
        for method,curve in s['curves'].items():
            for b in range(1,6):
                batch=[v for v in rows if v['method']==method and v['budget']==b]
                assert len(batch)==len({v['id'] for v in batch})==60
                assert all(len(set(v['checks']))==b for v in batch)
                assert sum(v['correct'] for v in batch)==curve[b-1]
        values=[f for f in features if f['kind']=='value'];total=sum(f['standardized_mse']*f['n'] for f in values)
        top=sorted(values,key=lambda f:f['standardized_mse']*f['n'],reverse=True)[:5]
        result[str(seed)]={'mean_budget_1_to_4':{m:sum(c[:4])/240 for m,c in s['curves'].items()},'curves':s['curves'],'top_features':top,'top5_error_fraction':sum(f['standardized_mse']*f['n'] for f in top)/total,'seconds':s['seconds'],'verified':True}
    (root/'aggregate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
