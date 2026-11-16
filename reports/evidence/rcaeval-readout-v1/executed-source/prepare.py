"""Fetch pinned RE1-OB data; split whole incidents and export blind arrays."""
import concurrent.futures, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
REV="afeacb11bcc94dadfd1c8f483ee4377b2b8b614e"
BASE=f"https://huggingface.co/datasets/phamquiluan/RCAEval/resolve/{REV}/"

def main():
    root=Path("data/rcaeval-v1");root.mkdir(parents=True,exist_ok=True)
    def fetch(name):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():
            resp=requests.get(BASE+name,timeout=90);resp.raise_for_status();p.write_bytes(resp.content)
        return name,hashlib.sha256(p.read_bytes()).hexdigest()
    hashes=dict([fetch("cases.parquet")]);idx=pd.read_parquet(root/"cases.parquet")
    rows=idx[idx.dataset=="RE1-OB"].sort_values("case").to_dict("records")
    assert len(rows)==125
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hashes.update(pool.map(fetch,[r["case"]+"/metrics.parquet" for r in rows]))
    blind=root/"blind";blind.mkdir(exist_ok=True);answers=[];manifest=[];excluded=[]
    for i,row in enumerate(rows):
        cid=f"incident-{i:04d}";df=pd.read_parquet(root/row["case"]/"metrics.parquet")
        assert "time" in df.columns
        df=df.sort_values("time");t=df.time.to_numpy(dtype=float)-float(row["inject_time"])
        keep=(t>=-300)&(t<300);cols=[c for c in df.columns if c!="time"]
        x=df.loc[keep,cols].to_numpy(dtype=np.float64);t=t[keep]
        if min((t<0).sum(),(t>=0).sum())<10:
            excluded.append(dict(case=row["case"],reason="Insufficient telemetry around published injection time",injection_time=int(row["inject_time"]),telemetry_start=float(df.time.min()),telemetry_end=float(df.time.max())))
            continue
        np.savez_compressed(blind/(cid+".npz"),time=t,values=x,columns=np.asarray(cols))
        split="train" if int(row["repetition"])<=3 else "validation" if int(row["repetition"])==4 else "test"
        manifest.append(dict(id=cid,split=split))
        answers.append(dict(id=cid,service=row["root_cause_service"],fault=row["fault"],split=split,source_case=row["case"]))
    (blind/"manifest.json").write_text(json.dumps(manifest,indent=2))
    (root/"answers.json").write_text(json.dumps(answers,indent=2))
    provenance=dict(dataset_revision=REV,dataset="RE1-OB",window_seconds=[-300,300],files=hashes,cases=len(manifest),excluded=excluded,
        split_rule="repetition 1-3 train; 4 validation; 5 test",license="MIT (dataset card)")
    (root/"provenance.json").write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(cases=len(manifest),excluded=excluded,bytes=sum(p.stat().st_size for p in blind.glob("*.npz")))))
if __name__=="__main__":main()
