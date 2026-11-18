"""Offline development analysis. Never collect data or update a checkpoint."""
import json
from pathlib import Path
import numpy as np
from microservice.regime import vector
from microservice.regime_evaluate import select,stats
from microservice.latent_probe import labels
from microservice.partial_readout import score,diagnose
from microservice.mechanisms import FAULT_MASK
R=Path(__file__).resolve().parents[1];E=R/'reports/evidence/lab-regime-v1'

from microservice.load_response import response

def main():
    rows={s:[json.loads(l) for l in (E/('collection/train.jsonl' if s=='train' else 'evaluation/validation.jsonl')).read_text().splitlines()] for s in ('train','validation')}
    saved=json.loads((E/'evaluation/validation-predictions.json').read_text());target=np.array(saved['target']);inverse={v:k for k,v in FAULT_MASK.items()}
    confusion={}
    for method in ('latent','history','latest'):
        matrix=np.zeros((8,8),int)
        for row,pred in zip(rows['validation'],saved[method]):matrix[row['fault'],inverse[pred]]+=1
        confusion[method]=matrix.tolist()
    x={s:np.array([response(r) for r in data]) for s,data in rows.items()}
    model,selection=select(x['train'],labels(rows['train']),x['validation'],target)
    pred=diagnose(score(model,x['validation']))
    # Scale using training only and compare validation's nearest same/different family.
    full={s:np.array([np.concatenate([vector(r,t) for t in range(4)]) for r in data]) for s,data in rows.items()}
    sd=full['train'].std(0);sd[sd<1e-6]=1
    distances=np.sqrt(np.mean(((full['validation'][:,None]-full['train'][None,:])/sd)**2,axis=-1))
    families=np.array([r['fault'] for r in rows['train']]);nearest=[]
    for i,row in enumerate(rows['validation']):
        same=np.where(families==row['fault'])[0];other=np.where(families!=row['fault'])[0];a=same[np.argmin(distances[i,same])];b=other[np.argmin(distances[i,other])]
        nearest.append(dict(episode=row['episode'],family=row['fault'],same_distance=float(distances[i,a]),other_distance=float(distances[i,b]),other_family=int(families[b]),other_closer=bool(distances[i,b]<distances[i,a])))
    ranges={}
    for f in range(8):
        ranges[str(f)]={s:{'ledger_log_latency_slope_quantiles':np.quantile(x[s][[r['fault']==f for r in rows[s]],10],[0,.25,.5,.75,1]).tolist()} for s in rows}
    result=dict(scope='Exploratory development analysis of previously used validation data, not an independent test or proof of identifiability.',confusion=confusion,load_response=dict(selection=selection,stats=stats(pred,rows['validation']),predictions=pred.tolist()),nearest_training_trace=nearest,slope_ranges=ranges)
    out=E/'development-analysis.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'load_response':result['load_response']['stats'],'nearest_wrong_family':sum(r['other_closer'] for r in nearest),'confusion':confusion},indent=2))
if __name__=='__main__':main()
