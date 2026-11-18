"""Descriptive error audit of a previously evaluated confirmation, never used for fitting."""
import json
from collections import Counter
import hashlib
import numpy as np
from scripts.develop_mechanism_readout import R, OUT

def main():
    path=R/'reports/evidence/lab-confirm-v1/evaluation/predictions.json'
    data=json.loads(path.read_text()); target=np.array(data['target']); methods={}
    for name in ('latent','response','response_forecast'):
        pred=np.array(data['methods'][name])
        pairs=Counter((int(t),int(p)) for t,p in zip(target,pred) if t!=p)
        methods[name]=dict(bit_errors_cpu_pool_delay=np.sum(((pred[:,None]>>np.arange(3))&1)!=((target[:,None]>>np.arange(3))&1),axis=0).tolist(),
            confusion=[dict(target_mask=t,predicted_mask=p,count=n) for (t,p),n in sorted(pairs.items())])
    result=dict(scope='Descriptive audit only; no fitting or model selection.',
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),methods=methods)
    (OUT/'confirmation-error-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
