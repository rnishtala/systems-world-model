"""Bounded, immutable training dataset uploads for the measured-probe protocol."""
import hashlib,json,math,os,re,shutil,time,uuid
from pathlib import Path

MAX_BYTES=2*1024*1024
MAX_ROWS=256
HELD_OUT_SEEDS={9202,9302,9401,9502,9503}
SERVICES={'checkout','payment','ledger'}

def number(value,low,high):
    return type(value) in (int,float) and math.isfinite(value) and low<=value<=high

def trajectory_hash(row):
    return hashlib.sha256(json.dumps({k:row[k] for k in ('base','extra','order')},sort_keys=True,separators=(',',':')).encode()).hexdigest()

def validate_jsonl(text,heldout=()):
    if not isinstance(text,str) or len(text.encode())>MAX_BYTES:raise ValueError('Choose a JSONL file no larger than 2 MiB')
    lines=text.splitlines()
    if not 64<=len(lines)<=MAX_ROWS:raise ValueError('Expected 64–256 incident rows, one JSON object per line')
    rows=[];seen=set();identities=set();coverage=set();counts={f:0 for f in range(8)}
    for line_no,line in enumerate(lines,1):
        try:
            row=json.loads(line)
            required={'seed','episode','fault','repeat','base','extra','order','seconds','utc'}
            if not isinstance(row,dict) or not required<=set(row) or set(row)-required-{'regime','split'}:raise ValueError('fields must match the incident format')
            for key,lo,hi in [('seed',1,999999999),('episode',0,1000000),('fault',0,7),('repeat',0,1000000)]:
                if type(row[key]) is not int or not lo<=row[key]<=hi:raise ValueError(f'invalid {key}')
            if 'regime' in row and (type(row['regime']) is not int or row['regime'] not in (0,1)):raise ValueError('invalid regime')
            if 'split' in row and row['split'] not in ('train','training','development'):raise ValueError('split must identify training/development data, not evaluation data')
            if row['seed'] in HELD_OUT_SEEDS:raise ValueError('seed belongs to a held-out evaluation dataset')
            if not number(row['seconds'],0,35) or not number(row['utc'],1,time.time()+300):raise ValueError('invalid collection duration or timestamp')
            if row['order'] not in (['checkout','ledger'],['ledger','checkout']):raise ValueError('order must contain checkout and ledger once')
            if not isinstance(row['base'],list) or len(row['base'])!=4:raise ValueError('base must contain four checkout windows')
            if not isinstance(row['extra'],dict) or set(row['extra'])!={'checkout','ledger'}:raise ValueError('extra requires checkout and ledger windows')
            windows=[(f,'checkout') for f in row['base']]
            for target in row['order']:
                if not isinstance(row['extra'][target],list) or len(row['extra'][target])!=2:raise ValueError('each extra target needs two windows')
                windows.extend((f,target) for f in row['extra'][target])
            for frame,target in windows:
                if not isinstance(frame,dict) or set(frame)!={'target','requests','successes','metrics','seconds'}:raise ValueError('invalid measurement window fields')
                if frame['target']!=target or type(frame['requests']) is not int or not 1<=frame['requests']<=12 or type(frame['successes']) is not int or frame['successes']!=frame['requests']:raise ValueError('windows require 1–12 successful measured requests at the correct target')
                if not number(frame['seconds'],0,35):raise ValueError('invalid window duration')
                metrics=frame['metrics']
                if not isinstance(metrics,dict) or set(metrics)!=SERVICES:raise ValueError('metrics require checkout, payment and ledger')
                for service,m in metrics.items():
                    if not isinstance(m,dict) or set(m)!={'count','latency_p95','self_p95','downstream_p95','error_rate'}:raise ValueError('invalid service metric fields')
                    expected=frame['requests'] if target=='checkout' or service=='ledger' else 0
                    if type(m['count']) is not int or m['count']!=expected:raise ValueError('metric counts do not match the measured requests')
                    if not all(number(m[k],0,35) for k in ('latency_p95','self_p95','downstream_p95')) or not number(m['error_rate'],0,0):raise ValueError('latencies must be finite seconds; measured requests must have zero errors')
            identity=(row['seed'],row['episode']);fingerprint=trajectory_hash(row)
            if identity in identities or fingerprint in seen:raise ValueError('duplicate incident or measurement trajectory')
            if fingerprint in heldout:raise ValueError('measurement trajectory overlaps a held-out evaluation incident')
            identities.add(identity);seen.add(fingerprint);rows.append(row)
            counts[row['fault']]+=1;coverage.add((row['fault'],row['repeat']%4))
        except (ValueError,TypeError,KeyError,OverflowError) as e:raise ValueError(f'Line {line_no}: {e}') from None
    if any(n<8 for n in counts.values()) or coverage!={(f,g) for f in range(8) for g in range(4)}:
        raise ValueError('Include all 8 fault families, at least 8 incidents per family, and all four repeat % 4 groups for each family')
    return rows

class Catalog:
    def __init__(self,root=None):self.root=Path(root or os.environ.get('RESULTS_ROOT','/results/experiments'))
    def all(self):
        from launcher.common import DATASETS
        result={k:dict(v,id=k,owner=None,seeds=[9101,9103,9501],source='built-in') for k,v in DATASETS.items()}
        for path in (self.root/'datasets').glob('upload-*.json'):
            d=json.loads(path.read_text());result[d['id']]=d
        return result
    def get(self,uid,email=None):
        from launcher.common import ADMIN
        if not isinstance(uid,str) or not re.fullmatch(r'expanded-128|upload-[a-f0-9]{24}',uid):raise ValueError('Choose an available training dataset')
        d=self.all().get(uid)
        if not d:raise ValueError('Training dataset not found')
        if email is not None and d['owner'] not in (None,email) and email!=ADMIN:raise PermissionError('This training dataset belongs to another user')
        return d
    def visible(self,email):
        from launcher.common import ADMIN
        return [self.public(d) for d in self.all().values() if d['owner'] in (None,email) or email==ADMIN]
    @staticmethod
    def public(d):return {k:d[k] for k in ('id','name','incidents','sha256','owner','source','seeds')}
    def heldout(self):
        # Retained tests remain protected after their worker Jobs are cleaned up.
        result=set()
        for path in (self.root/'runs').glob('*/test.jsonl'):
            for line in path.read_text().splitlines():
                try:result.add(trajectory_hash(json.loads(line)))
                except (ValueError,KeyError):pass  # An active collector may be appending its final line.
        return result
    def upload(self,owner,body):
        if not isinstance(body,dict) or set(body)!={'name','jsonl','training_only'}:raise ValueError('Expected dataset name, JSONL content and training-only confirmation')
        if not isinstance(body['name'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}',body['name']):raise ValueError('Use 1–64 letters, numbers, spaces, dots, underscores or hyphens for the dataset name')
        if body['training_only'] is not True:raise ValueError('Confirm that this file contains training incidents only')
        items=list(self.all().values())
        if sum(d['owner']==owner for d in items)>=5 or sum(d['owner'] is not None for d in items)>=20:raise ValueError('Dataset limit reached (5 per user, 20 for this lab)')
        rows=validate_jsonl(body['jsonl'],self.heldout())
        self.root.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(self.root).free<1024**3:raise ValueError('Less than 1 GiB of lab storage is free')
        content=('\n'.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False) for r in rows)+'\n').encode()
        digest=hashlib.sha256(content).hexdigest()
        for d in items:
            if d['owner']==owner and d['sha256']==digest:raise ValueError('You already uploaded this dataset: '+d['name'])
        uid='upload-'+uuid.uuid4().hex[:24];folder=self.root/'datasets';folder.mkdir(exist_ok=True)
        data=folder/(uid+'.jsonl');meta=folder/(uid+'.json')
        d=dict(id=uid,name=body['name'],incidents=len(rows),sha256=digest,owner=owner,seeds=sorted({r['seed'] for r in rows}),source='uploaded',created=time.time())
        with data.open('xb') as f:f.write(content);f.flush();os.fsync(f.fileno())
        temporary=meta.with_suffix('.tmp')
        with temporary.open('x') as f:json.dump(d,f);f.flush();os.fsync(f.fileno())
        temporary.replace(meta)  # Publish only after the immutable data is durable.
        return self.public(d)
    def verify(self,uid,expected=None):
        d=self.get(uid)
        path=Path(d['path']) if d['source']=='built-in' else self.root/'datasets'/(uid+'.jsonl')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=(expected or d['sha256']):raise ValueError('Training dataset hash mismatch')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        if len(rows)!=d['incidents'] or sorted({r['seed'] for r in rows})!=d['seeds']:raise ValueError('Training split changed')
        return path
