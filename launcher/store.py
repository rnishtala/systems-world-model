"""Durable FIFO queue. One active experiment across launcher restarts."""
import json,secrets,sqlite3,time,uuid
from launcher.common import ACTIVE,TERMINAL,validate_request,validate_expansion,validate_retest

class Store:
    def __init__(self,path):
        self.db=sqlite3.connect(path);self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, created REAL NOT NULL, body TEXT NOT NULL)');self.db.commit()
    def all(self):return [json.loads(x[0]) for x in self.db.execute('SELECT body FROM runs ORDER BY created')]
    def get(self,uid):
        row=self.db.execute('SELECT body FROM runs WHERE id=?',(uid,)).fetchone()
        if row is None:raise KeyError(uid)
        return json.loads(row[0])
    def save(self,run):
        self.db.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?)',(run['id'],run['created'],json.dumps(run)));self.db.commit()
    def create(self,owner,body,image,version,dataset=None,expansion=None,evaluation=None):
        args=validate_request(body,{dataset['id']:dataset} if dataset else None);existing=self.all();pending=[r for r in existing if r['status'] not in TERMINAL]
        if len(pending)>=20 or sum(r['owner']==owner for r in pending)>=3:raise ValueError('Queue limit reached; wait for an existing run')
        used={r['test_seed'] for r in existing}|set(dataset['seeds'] if dataset else []);seed=1000000000+secrets.randbelow(1000000000)
        while seed in used:seed=1000000000+secrets.randbelow(1000000000)
        run=dict(**args,id=uuid.uuid4().hex[:12],owner=owner,created=time.time(),status='queued',phase='queued',test_seed=seed,image=image,version=version,cancel=False,error=None)
        if evaluation is not None:
            if expansion is not None:raise ValueError('Frozen test cannot expand training')
            validate_retest(evaluation['candidate_run'],evaluation['baseline_run'])
            for source in (evaluation['candidate_run'],evaluation['baseline_run']):
                previous=self.get(source)
                if previous['status']!='completed' or previous['owner']!=owner:raise ValueError('Use completed models owned by this user')
            candidate=self.get(evaluation['candidate_run'])
            if candidate['seed']!=args['seed'] or candidate['steps']!=args['steps']:raise ValueError('Candidate settings mismatch')
            run.update(mode='test-only',evaluate_run=evaluation['candidate_run'],baseline_run=evaluation['baseline_run'])
        if expansion is not None:
            validate_expansion(expansion['seed'],expansion['baseline_run'],seed)
            if expansion['seed'] in used:raise ValueError('Expansion seed overlaps a known seed')
            run.update(expansion_seed=expansion['seed'],baseline_run=expansion['baseline_run'])
        if dataset:run.update(dataset_sha256=dataset['sha256'],dataset_name=dataset['name'],training_seeds=dataset['seeds'])
        self.save(run);return run
