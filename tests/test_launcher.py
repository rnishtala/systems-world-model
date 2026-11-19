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
