"""Read-only, bounded-chunk recovery of completed experiment evidence."""
import base64,hashlib,json,shlex,subprocess,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R=Path(__file__).resolve().parents[1];OUT=R/'reports/evidence/probe-live-budget-v1'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','rajnishtala@amd-halo']
K='kubectl --kubeconfig=$HOME/.kube/graphrobin.yaml -n systems-world-model-lab '
POD='swm-probe-live-budget-export-v1';REMOTE='/results/probe-live-budget-v1'
def run(cmd,input=None,retries=4):
    for attempt in range(retries):
        try:
            p=subprocess.run(SSH+[K+cmd],input=input,capture_output=True,text=True,timeout=60)
            if p.returncode:raise RuntimeError(p.stderr[-1500:])
            return p.stdout
        except (RuntimeError,subprocess.TimeoutExpired):
            if attempt+1==retries:raise
            time.sleep(2)
def python(code):return run('exec '+POD+' -- /results/venv/bin/python -c '+shlex.quote(code))
def main():
    pod=json.loads((R/'deploy/rcaeval-stage-v1.json').read_text());pod['metadata']['name']=POD
    container=pod['spec']['containers'][0];container['command']=['sleep','1800']
    for mount in container['volumeMounts']:
        if mount['name']=='results':mount['readOnly']=True
    run('create -f -',json.dumps(pod),1)
    try:
        run('wait pod/'+POD+' --for=condition=Ready --timeout=45s')
        code="from pathlib import Path; import json,hashlib; r=Path("+repr(REMOTE)+"); paths=list(r.glob('frozen/*.json'))+list(r.glob('frozen/*.npz'))+list(r.glob('collection/*.json*'))+list(r.glob('evaluation/*.json*'))+list(r.glob('evaluation/*.npz'))+list(r.glob('models/*.json')); print(json.dumps({str(p.relative_to(r)):{'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths}))"
        manifest=json.loads(python(code));assert manifest
        for name,meta in manifest.items():
            path=Path(name);assert not path.is_absolute() and '..' not in path.parts
            dest=OUT/path
            if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==meta['sha256']:continue
            data=bytearray()
            def chunk(offset):
                command='dd if='+shlex.quote(REMOTE+'/'+name)+' bs=32768 skip='+str(offset//32768)+' count=1 status=none | base64 -w0'
                encoded=run('exec '+POD+' -- sh -c '+shlex.quote(command)).strip()
                decoded=base64.b64decode(encoded,validate=True)
                assert len(decoded)==min(32768,meta['size']-offset)
                return decoded
            with ThreadPoolExecutor(max_workers=3) as pool:
                for decoded in pool.map(chunk,range(0,meta['size'],32768)):data.extend(decoded)
            assert len(data)==meta['size'] and hashlib.sha256(data).hexdigest()==meta['sha256'],name
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
            print('EXPORTED',name,meta['size'],flush=True)
        code="from pathlib import Path; import json,hashlib; r=Path("+repr(REMOTE)+"); print(json.dumps({str(p.relative_to(Path('/results/probe-target-model-v1'))):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/results/probe-target-model-v1').glob('models/final-91/**/*') if p.is_file()}))"
        (OUT/'checkpoint-hashes.json').write_text(json.dumps(json.loads(python(code)),indent=2)+'\n')
        (OUT/'frozen-hashes.json').write_text(json.dumps({n:m['sha256'] for n,m in manifest.items()},indent=2)+'\n')
        (OUT/'export-recovery.json').write_text(json.dumps(dict(read_only_pvc=True,chunk_bytes=32768,parallel_readers=3,files=len(manifest),all_hashes_verified=True),indent=2)+'\n')
    finally:run('delete pod '+POD+' --grace-period=1 --wait=true --timeout=45s --ignore-not-found=true')
if __name__=='__main__':main()
