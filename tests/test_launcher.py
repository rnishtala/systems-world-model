import asyncio, copy, json, time
from unittest.mock import patch
import pytest
from aiohttp.test_utils import TestClient, TestServer
from launcher.common import validate_request, job_manifest, ADMIN, ORIGIN
from launcher.auth import signature
from launcher.store import Store
from launcher.server import create_app
from launcher.controller import Controller

IMAGE='ghcr.io/rnishtala/systems-world-model-runner@sha256:'+'a'*64
ARGS=dict(name='Test',dataset='expanded-128',seed=112,steps=1000)

@pytest.mark.parametrize('field,value',[('dataset',[]),('steps',True),('seed',0),('name','../escape'),('command','sh')])
def test_reject_unapproved_input(field,value):
    with pytest.raises(ValueError):validate_request(dict(ARGS,**{field:value}))

def test_manifest_boundaries():
    r=dict(ARGS,id='a'*12,test_seed=123)
    m=job_manifest(r,IMAGE);p=m['spec']['template']['spec']
    assert m['metadata']['namespace']=='systems-world-model-lab'
    assert p['automountServiceAccountToken'] is False
    assert p['containers'][0]['command'][:3]==['python','-m','launcher.pipeline']
    with pytest.raises(ValueError):job_manifest(r,'arbitrary:latest')

def test_queue_persistence_limits(tmp_path):
    s=Store(tmp_path/'queue');runs=[s.create('a@example.com',ARGS,IMAGE,'test') for _ in range(3)]
    with pytest.raises(ValueError):s.create('a@example.com',ARGS,IMAGE,'test')
    assert Store(tmp_path/'queue').all()==runs
    assert len({r['test_seed'] for r in runs})==3

def test_identity_ownership_and_csrf(tmp_path,monkeypatch):
    monkeypatch.setenv('RUNNER_IMAGE',IMAGE);monkeypatch.setenv('CODE_VERSION','test')
    async def check():
        s=Store(tmp_path/'queue');key=b'test-key'
        async with TestClient(TestServer(create_app(s,tmp_path,key))) as client:
            async def req(method,path,email='a@example.com',body=None,origin=ORIGIN,stamp=None,signed=True):
                raw=json.dumps(body).encode() if body is not None else b'';stamp=stamp or str(time.time())
                headers={'X-Swm-Authenticated-Email':email,'X-Swm-Timestamp':stamp,'Origin':origin,'Content-Type':'application/json'}
                if signed:headers['X-Swm-Signature']=signature(key,method,path,email,stamp,raw)
                return await client.request(method,path,data=raw,headers=headers)
            assert (await req('GET','/api/runs',signed=False)).status==403
            assert (await req('GET','/api/runs',stamp='1')).status==403
            assert (await req('POST','/api/runs',body=ARGS,origin='https://evil.example')).status==403
            response=await req('POST','/api/runs',body=ARGS);assert response.status==201
            run=await response.json();path='/api/runs/'+run['id']
            assert 'restore_spec' not in run
            assert (await req('GET',path,email='b@example.com')).status==403
            assert (await req('POST',path+'/cancel',email='b@example.com',body={})).status==403
            assert await (await req('GET','/api/runs',email='b@example.com')).json()==[]
            assert (await req('GET',path,email=ADMIN)).status==200
            assert (await req('POST',path+'/cancel',email=ADMIN,body={})).status==200
            assert s.get(run['id'])['status']=='cancelled'
    asyncio.run(check())

class FakeKube:
    def __init__(self):
        self.dep={'spec':{'replicas':1,'template':{'unchanged':True}},'status':{'readyReplicas':1}}
        self.job=None;self.scales=[];self.creates=0
    async def jobs(self):return [self.job] if self.job else []
    async def deployment(self):return copy.deepcopy(self.dep)
    async def scale(self,n):self.scales.append(n);self.dep['spec']['replicas']=n;self.dep['status']['readyReplicas']=n
    async def quota(self):return {'status':{'used':{'limits.cpu':'3995m' if self.job else '1995m'}}}
    async def create(self,m):self.creates+=1;self.job=m;return m
    async def delete(self,uid):self.job=None
    async def pods(self,uid):return []

@pytest.mark.parametrize('outcome',['completed','cancelled','failed','restart'])
def test_controller_restore_and_recover(tmp_path,outcome):
    async def check():
        s=Store(tmp_path/'queue');r=s.create('a@example.com',ARGS,IMAGE,'test');k=FakeKube();c=Controller(s,k,tmp_path)
        with patch('launcher.controller.dataset_check'):
            await c.tick()
        assert s.get(r['id'])['status']=='running';assert k.scales==[0]
        if outcome=='restart':
            r=s.get(r['id']);r['status']='preparing';s.save(r)
            await Controller(s,k,tmp_path).tick()
            assert s.get(r['id'])['status']=='running';assert k.creates==1
        elif outcome=='cancelled':
            r=s.get(r['id']);r['cancel']=True;s.save(r)
        k.job['status']={'failed':1} if outcome=='failed' else {'succeeded':1}
        await c.tick()
        r=s.get(r['id']);r['release_after']=0;s.save(r)
        await c.tick()
        assert s.get(r['id'])['status']==('completed' if outcome=='restart' else outcome)
        assert k.scales==[0,1];assert k.dep['spec']['template']=={'unchanged':True}
    asyncio.run(check())


def test_manifest_resources_stay_in_namespace():
    from launcher.manifests import resources
    images=dict(control=IMAGE.replace('-runner@','-control@'),runner=IMAGE,version='test')
    items=resources(images)['items']
    assert all(i['metadata']['namespace']=='systems-world-model-lab' for i in items)
    assert not any(i['kind'].startswith('Cluster') for i in items)
    role=next(i for i in items if i['kind']=='Role')
    assert all('secrets' not in r['resources'] for r in role['rules'])
    scale=next(r for r in role['rules'] if r['resources']==['deployments/scale'])
    assert scale['resourceNames']==['swm-inference']

def test_controller_preserves_unrelated_inference_edits(tmp_path):
    async def check():
        s=Store(tmp_path/'queue');r=s.create('a@example.com',ARGS,IMAGE,'test');k=FakeKube();c=Controller(s,k,tmp_path)
        with patch('launcher.controller.dataset_check'):await c.tick()
        k.dep['spec']['template']['changed']=True;k.job['status']={'succeeded':1}
        await c.tick()
        assert s.get(r['id'])['status']=='restoring'
        assert 'refusing to overwrite' in s.get(r['id'])['error']
        assert k.scales==[0]
    asyncio.run(check())


def test_proxy_preserves_compression_headers():
    from launcher.proxy import HOP
    assert 'content-encoding' not in HOP
    assert 'transfer-encoding' in HOP


@pytest.mark.parametrize('seed,baseline',[(True,'a'*12),(0,'a'*12),(1000000000,'a'*12),(9601,'../baseline')])
def test_expansion_rejects_invalid_protocol(seed,baseline):
    from launcher.common import validate_expansion
    with pytest.raises(ValueError):validate_expansion(seed,baseline,1461433385)


def test_expansion_is_durable_and_scoped(tmp_path):
    store=Store(tmp_path/'queue')
    run=store.create(ADMIN,dict(ARGS,seed=113,steps=100000),IMAGE,'test',expansion=dict(seed=9601,baseline_run='f4405906195f'))
    saved=Store(tmp_path/'queue').get(run['id'])
    assert saved['expansion_seed']==9601
    command=job_manifest(saved,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert command[-4:]==['--expand-training-seed','9601','--baseline-run','f4405906195f']
    # Public requests cannot smuggle an internal experimental protocol.
    with pytest.raises(ValueError):validate_request(dict(ARGS,expansion_seed=9601))


def test_frozen_test_requires_completed_owned_sources(tmp_path):
    store=Store(tmp_path/'queue')
    candidate=store.create(ADMIN,ARGS,IMAGE,'test')
    baseline=store.create(ADMIN,ARGS,IMAGE,'test')
    protocol=dict(candidate_run=candidate['id'],baseline_run=baseline['id'])
    with pytest.raises(ValueError):store.create(ADMIN,ARGS,IMAGE,'test',evaluation=protocol)
    for r in (candidate,baseline):r['status']='completed';store.save(r)
    run=store.create(ADMIN,ARGS,IMAGE,'test',evaluation=protocol)
    command=job_manifest(run,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert command[:3]==['python','-m','launcher.retest']
    assert command[-4:]==['--candidate-run',candidate['id'],'--baseline-run',baseline['id']]
    assert run['test_seed'] not in (candidate['test_seed'],baseline['test_seed'])
    with pytest.raises(ValueError):store.create('other@example.com',ARGS,IMAGE,'test',evaluation=protocol)
    with pytest.raises(ValueError):store.create(ADMIN,dict(ARGS,seed=113),IMAGE,'test',evaluation=protocol)
    with pytest.raises(ValueError):store.create(ADMIN,ARGS,IMAGE,'test',evaluation=protocol,expansion=dict(seed=9601,baseline_run=baseline['id']))


@pytest.mark.parametrize('candidate,baseline',[('../models','a'*12),('a'*12,'a'*12),('a'*12,None)])
def test_frozen_test_invalid_sources(candidate,baseline):
    from launcher.common import validate_retest
    with pytest.raises(ValueError):validate_retest(candidate,baseline)


def test_training_comparison_pins_source_and_settings(tmp_path):
    s=Store(tmp_path/'queue');source=s.create(ADMIN,ARGS,IMAGE,'old')
    source['status']='completed';s.save(source)
    r=s.create(ADMIN,ARGS,IMAGE,'new',training_source=source['id'])
    assert r['baseline_run']==source['id']
    command=job_manifest(r,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert command[-2:]==['--training-source-run',source['id']]
    with pytest.raises(ValueError):s.create('other@example.com',ARGS,IMAGE,'new',training_source=source['id'])
    with pytest.raises(ValueError):s.create(ADMIN,dict(ARGS,seed=114),IMAGE,'new',training_source=source['id'])
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',training_source='../escape')


@pytest.mark.parametrize('seeds',[[1,2],[1,1,2],[True,2,3],[0,2,3],[1000000000,2,3],'123'])
def test_probe_protocol_rejects_invalid_seeds(seeds):
    from launcher.common import validate_probe_seeds
    with pytest.raises(ValueError):validate_probe_seeds(seeds)


def test_probe_collection_is_durable_bounded_and_operator_only(tmp_path):
    from launcher.common import run_budget
    s=Store(tmp_path/'queue');source=s.create(ADMIN,ARGS,IMAGE,'old')
    source['status']='completed';s.save(source)
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',probe_training_seeds=[9601,9602,9603])
    run=s.create(ADMIN,ARGS,IMAGE,'new',training_source=source['id'],probe_training_seeds=[9601,9602,9603])
    saved=Store(tmp_path/'queue').get(run['id'])
    assert saved['training_seeds']==[9601,9602,9603]
    assert saved['probe_schedule']=='ledger-first'
    job=job_manifest(saved,IMAGE)
    assert job['spec']['activeDeadlineSeconds']==run_budget(saved)==7200
    assert run_budget(source)==3600
    cmd=job['spec']['template']['spec']['containers'][0]['command']
    assert cmd[-4:]==['--probe-training-seeds','9601','9602','9603']
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',training_source=source['id'],probe_training_seeds=[9601,9604,9605])
    with pytest.raises(ValueError):validate_request(dict(ARGS,probe_training_seeds=[9601,9602,9603]))


def test_forecast_uses_completed_owned_model_and_separate_worker(tmp_path):
    s=Store(tmp_path/'queue');source=s.create(ADMIN,ARGS,IMAGE,'old')
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',forecast_source=source['id'])
    source['status']='completed';s.save(source)
    run=s.create(ADMIN,ARGS,IMAGE,'new',forecast_source=source['id'])
    cmd=job_manifest(run,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert cmd[:3]==['python','-m','launcher.forecast'] and run['mode']=='forecast'
    assert cmd[-2:]==['--candidate-run',source['id']]
    with pytest.raises(ValueError):s.create('other@example.com',ARGS,IMAGE,'new',forecast_source=source['id'])
    with pytest.raises(ValueError):s.create(ADMIN,dict(ARGS,seed=999),IMAGE,'new',forecast_source=source['id'])
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',forecast_source=source['id'],training_source=source['id'])
    with pytest.raises(ValueError):validate_request(dict(ARGS,forecast_source=source['id']))


def test_forecast_diagnostic_requires_matching_completed_owned_forecast(tmp_path):
    s=Store(str(tmp_path/'audit.db'))
    source=s.create(ADMIN,ARGS,IMAGE,'old');source['status']='completed';s.save(source)
    prior=s.create(ADMIN,ARGS,IMAGE,'old',forecast_source=source['id'])
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',forecast_source=source['id'],audit_source=prior['id'])
    prior['status']='completed';s.save(prior)
    run=s.create(ADMIN,ARGS,IMAGE,'new',forecast_source=source['id'],audit_source=prior['id'])
    cmd=job_manifest(run,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert cmd[-2:]==['--audit-source',prior['id']]
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'new',audit_source=prior['id'])
    with pytest.raises(ValueError):validate_request(dict(ARGS,audit_source=prior['id']))


def test_triage_requires_retained_forecast_and_uses_bounded_worker(tmp_path):
    s=Store(str(tmp_path/'triage.db'))
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'v',triage=True)
    source=s.create(ADMIN,ARGS,IMAGE,'v');source['status']='completed';s.save(source)
    prior=s.create(ADMIN,ARGS,IMAGE,'v',forecast_source=source['id']);prior['status']='completed';s.save(prior)
    run=s.create(ADMIN,ARGS,IMAGE,'v',forecast_source=source['id'],audit_source=prior['id'],triage=True)
    job=job_manifest(run,IMAGE)
    assert job['spec']['template']['spec']['containers'][0]['command'][-1]=='--triage'
    assert job['spec']['activeDeadlineSeconds']==3600
    with pytest.raises(ValueError):validate_request(dict(ARGS,triage=True))


def test_fresh_triage_cannot_reuse_retained_tests(tmp_path):
    s=Store(str(tmp_path/'fresh.db'))
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'v',fresh_triage=True)
    source=s.create(ADMIN,ARGS,IMAGE,'v');source['status']='completed';s.save(source)
    run=s.create(ADMIN,ARGS,IMAGE,'v',forecast_source=source['id'],fresh_triage=True)
    assert run['fresh_triage'] and run['triage'] and not run.get('audit_source')
    cmd=job_manifest(run,IMAGE)['spec']['template']['spec']['containers'][0]['command']
    assert '--fresh-triage' in cmd and '--audit-source' not in cmd
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'v',forecast_source=source['id'],fresh_triage=True,triage=True)
    with pytest.raises(ValueError):validate_request(dict(ARGS,fresh_triage=True))


def test_temporal_pilot_is_separate_and_namespace_bounded(tmp_path):
    s=Store(str(tmp_path/'temporal.db'))
    run=s.create(ADMIN,ARGS,IMAGE,'v',temporal_pilot=True)
    job=job_manifest(run,IMAGE)
    assert job['metadata']['namespace']=='systems-world-model-lab'
    assert job['spec']['template']['spec']['containers'][0]['command'][2]=='launcher.temporal_pipeline'
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'v',temporal_pilot=True,fresh_triage=True)
    with pytest.raises(ValueError):validate_request(dict(ARGS,temporal_pilot=True))


def test_snapshot_calibration_requires_temporal_mode(tmp_path):
    s=Store(tmp_path/'queue.sqlite')
    with pytest.raises(ValueError):s.create(ADMIN,ARGS,IMAGE,'v',snapshot_calibration=True)
    run=s.create(ADMIN,ARGS,IMAGE,'v',temporal_pilot=True,snapshot_calibration=True)
    assert '--snapshot-calibration' in job_manifest(run,IMAGE)['spec']['template']['spec']['containers'][0]['command']
