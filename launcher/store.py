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
    def create(self,owner,body,image,version,dataset=None,expansion=None,evaluation=None,training_source=None,probe_training_seeds=None,forecast_source=None,audit_source=None,triage=False,fresh_triage=False,temporal_pilot=False,snapshot_calibration=False,action_outcomes=False):
        args=validate_request(body,{dataset['id']:dataset} if dataset else None);existing=self.all();pending=[r for r in existing if r['status'] not in TERMINAL]
        if len(pending)>=20 or sum(r['owner']==owner for r in pending)>=3:raise ValueError('Queue limit reached; wait for an existing run')
        used={r['test_seed'] for r in existing}|set(dataset['seeds'] if dataset else []);seed=1000000000+secrets.randbelow(1000000000)
        while seed in used:seed=1000000000+secrets.randbelow(1000000000)
        run=dict(**args,id=uuid.uuid4().hex[:12],owner=owner,created=time.time(),status='queued',phase='queued',test_seed=seed,image=image,version=version,cancel=False,error=None)
        if action_outcomes and (not temporal_pilot or snapshot_calibration):raise ValueError('Action outcomes requires a separate temporal pilot')
        if action_outcomes:run['action_outcomes']=True
        if snapshot_calibration and not temporal_pilot:raise ValueError('Calibration requires temporal pilot')
        if snapshot_calibration:run['snapshot_calibration']=True
        if temporal_pilot:
            if any(v is not None for v in (expansion,evaluation,training_source,probe_training_seeds,forecast_source,audit_source)) or triage or fresh_triage:raise ValueError('Temporal pilot must run alone')
            run.update(mode='temporal-pilot',temporal_pilot=True)
        if forecast_source is not None:
            if any(v is not None for v in (training_source,probe_training_seeds,expansion,evaluation)):raise ValueError('Forecast experiment must run alone')
            validate_retest(run['id'],forecast_source)
            source=self.get(forecast_source)
            if source['status']!='completed' or source['owner']!=owner:raise ValueError('Use a completed owned source model')
            if any(source[k]!=args[k] for k in ('seed','steps','dataset')):raise ValueError('Source settings mismatch')
            run.update(mode='forecast',forecast_source=forecast_source,probe_schedule=source.get('probe_schedule','mixed'))
        if audit_source is not None:
            if forecast_source is None:raise ValueError('Diagnostic requires a forecast model')
            validate_retest(run['id'],audit_source)
            previous=self.get(audit_source)
            if previous['status']!='completed' or previous['owner']!=owner or previous.get('forecast_source')!=forecast_source or previous.get('audit_source'):raise ValueError('Use a completed owned forecast from the same model')
            run.update(audit_source=audit_source)
        if fresh_triage:
            if forecast_source is None or audit_source is not None or triage:raise ValueError('Fresh triage requires only a frozen source model')
            run.update(fresh_triage=True,triage=True)
        if triage:
            if audit_source is None:raise ValueError('Triage requires retained forecast data')
            run.update(triage=True)
        if training_source is not None:
            if expansion is not None or evaluation is not None:raise ValueError('Training comparison must run alone')
            import re
            if not isinstance(training_source,str) or not re.fullmatch('[a-f0-9]{12}',training_source):raise ValueError('Invalid training source')
            previous=self.get(training_source)
            if previous['status']!='completed' or previous['owner']!=owner:raise ValueError('Use a completed model owned by this user')
            if any(previous[k]!=args[k] for k in ('dataset','seed','steps')):raise ValueError('Keep source dataset, seed and steps fixed')
            run.update(training_source=training_source,baseline_run=training_source)
        if probe_training_seeds is not None:
            from launcher.common import validate_probe_seeds
            seeds=validate_probe_seeds(probe_training_seeds)
            if training_source is None:raise ValueError('Probe experiment requires a frozen reference')
            known=used|{v for r in existing for v in r.get('probe_training_seeds',[])}
            if set(seeds)&known:raise ValueError('Probe collection seed already used')
            run.update(probe_training_seeds=seeds,probe_schedule='ledger-first')
        if evaluation is not None:
            if expansion is not None:raise ValueError('Frozen test cannot expand training')
            validate_retest(evaluation['candidate_run'],evaluation['baseline_run'])
            for source in (evaluation['candidate_run'],evaluation['baseline_run']):
                previous=self.get(source)
                if previous['status']!='completed' or previous['owner']!=owner:raise ValueError('Use completed models owned by this user')
            candidate=self.get(evaluation['candidate_run'])
            if candidate['seed']!=args['seed'] or candidate['steps']!=args['steps']:raise ValueError('Candidate settings mismatch')
            run.update(probe_schedule=candidate.get('probe_schedule','mixed'),mode='test-only',evaluate_run=evaluation['candidate_run'],baseline_run=evaluation['baseline_run'])
        if expansion is not None:
            validate_expansion(expansion['seed'],expansion['baseline_run'],seed)
            if expansion['seed'] in used:raise ValueError('Expansion seed overlaps a known seed')
            run.update(expansion_seed=expansion['seed'],baseline_run=expansion['baseline_run'])
        if dataset:run.update(dataset_sha256=dataset['sha256'],dataset_name=dataset['name'],training_seeds=dataset['seeds'])
        if probe_training_seeds is not None:run.update(training_seeds=seeds,dataset_name='Fresh ledger-first training - 192 incidents')
        self.save(run);return run
