"""Read-only comparison; readiness/restarts cannot establish zero latency impact."""
import json,sys

def inventory(path):
    d=json.load(open(path));out={}
    for p in d['items']:
        if p['metadata']['namespace']=='systems-world-model-lab':continue
        if p['status'].get('phase')!='Running':continue
        key=p['metadata']['namespace']+'/'+p['metadata']['name']
        cs=p['status'].get('containerStatuses',[])
        out[key]={'ready':all(c.get('ready',False) for c in cs),
                  'restarts':sum(c.get('restartCount',0) for c in cs)}
    return out
before,after=map(inventory,sys.argv[1:3]);common=set(before)&set(after)
print(json.dumps({'before_running':len(before),'after_running':len(after),
 'after_unready':sorted(k for k,v in after.items() if not v['ready']),
 'readiness_regressions':[k for k in common if before[k]['ready'] and not after[k]['ready']],
 'restart_increases':[k for k in common if after[k]['restarts']>before[k]['restarts']],
 'missing_pods':sorted(set(before)-set(after)), 'new_pods':sorted(set(after)-set(before)),
 'limitation':'Readiness and restart checks do not measure application latency or prove zero shared-host interference.'},indent=2))
