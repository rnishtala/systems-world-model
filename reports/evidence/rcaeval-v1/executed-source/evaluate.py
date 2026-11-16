"""Blind metric ranking; labels are accessed only by the separate scoring call."""
import argparse,json,time
from pathlib import Path
import numpy as np

def rank_case(path):
    with np.load(path,allow_pickle=False) as d:
        t,x,columns=d["time"],d["values"],d["columns"].tolist()
    scores=[]
    for j,col in enumerate(columns):
        a=x[t<0,j];b=x[t>=0,j];a=a[np.isfinite(a)];b=b[np.isfinite(b)]
        if len(a)<2 or len(b)<2 or np.ptp(a)==0 or np.ptp(b)==0:continue
        # BARO-style robust scaling, positive maximum shift, constant columns removed.
        scale=float(np.percentile(a,75)-np.percentile(a,25))
        if scale<10*np.finfo(float).eps:scale=1.
        scores.append((col,float(np.max((b-np.median(a))/scale))))
    scores.sort(key=lambda item:(-item[1],item[0]))
    services=list(dict.fromkeys(col.split("_",1)[0] for col,_ in scores))
    return services,scores

def score(predictions,answers):
    truth={r["id"]:r for r in answers}
    if set(truth)!={r["id"] for r in predictions} or len(predictions)!=len(truth):raise ValueError("Case coverage mismatch")
    result={}
    for split in ("train","validation","test","all"):
        rows=[r for r in predictions if split=="all" or truth[r["id"]]["split"]==split]
        acc={str(k):sum(truth[r["id"]]["service"] in r["ranks"][:k] for r in rows)/len(rows) for k in range(1,6)}
        faults={f:{"n":sum(truth[r["id"]]["fault"]==f for r in rows),"top1_correct":sum(truth[r["id"]]["fault"]==f and truth[r["id"]]["service"] in r["ranks"][:1] for r in rows)} for f in sorted({truth[r["id"]]["fault"] for r in rows})}
        result[split]=dict(n=len(rows),ac_at_k=acc,avg_at_5=sum(acc.values())/5,faults=faults)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--out",required=True);p.add_argument("--answers");a=p.parse_args()
    root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rows=[];started=time.monotonic()
    for r in json.loads((root/"manifest.json").read_text()):
        start=time.monotonic();ranks,scores=rank_case(root/(r["id"]+".npz"))
        rows.append(dict(id=r["id"],ranks=ranks,indicator_scores=scores,seconds=time.monotonic()-start))
    (out/"predictions.json").write_text(json.dumps(rows,indent=2))
    summary=dict(method="baro_style_raw_metrics",scope="Local BARO-style comparator; not exact published preprocessing or paper reproduction",seconds=time.monotonic()-started)
    if a.answers:summary["scores"]=score(rows,json.loads(Path(a.answers).read_text()))
    (out/"summary.json").write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=="__main__":main()
