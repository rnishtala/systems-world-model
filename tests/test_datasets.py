import asyncio,copy,json,time
from pathlib import Path
import pytest
from aiohttp.test_utils import TestClient,TestServer
from launcher.auth import signature
from launcher.common import ADMIN,ORIGIN,job_manifest
from launcher.datasets import Catalog,validate_jsonl,trajectory_hash,MAX_BYTES
from launcher.server import create_app
from launcher.store import Store
IMAGE='ghcr.io/rnishtala/systems-world-model-runner@sha256:'+'a'*64
ARGS=dict(name='Test',dataset='expanded-128',seed=112,steps=1000)

@pytest.fixture
def incidents():
    rows=[]
    for fault in range(8):
        for repeat in range(8):
            episode=len(rows)
            def frame(target):
                return dict(target=target,requests=3,successes=3,seconds=.1,metrics={s:dict(count=3 if target=='checkout' or s=='ledger' else 0,latency_p95=.01+episode*.0001,self_p95=.005,downstream_p95=.005,error_rate=0) for s in ('checkout','payment','ledger')})
            rows.append(dict(seed=1234,episode=episode,fault=fault,repeat=repeat,regime=repeat//4,base=[frame('checkout') for _ in range(4)],extra={t:[frame(t) for _ in range(2)] for t in ('checkout','ledger')},order=['checkout','ledger'],seconds=1.,utc=time.time()-100))
    return rows

def encode(rows):return '\n'.join(json.dumps(r) for r in rows)+'\n'
def body(rows):return dict(name='Training upload',jsonl=encode(rows),training_only=True)

@pytest.mark.parametrize('case',['nan','negative','count','failed','frame_count','service','label','coverage','duplicate','test_seed','evaluation_seed','future','unknown'])
def test_reject_incompatible_data(incidents,case):
    rows=incidents;r=rows[0];m=r['base'][0]['metrics']['ledger']
    if case=='nan':m['latency_p95']=float('nan')
    if case=='negative':m['latency_p95']=-1
    if case=='count':m['count']=1
    if case=='failed':r['base'][0]['successes']=2
    if case=='frame_count':r['base'].pop()
    if case=='service':r['extra']['other']=r['extra'].pop('ledger')
    if case=='label':r['fault']=8
    if case=='coverage':
        for x in rows:x['fault']=0
    if case=='duplicate':rows[1]=copy.deepcopy(rows[0])
    if case=='test_seed':r['seed']=9503
    if case=='evaluation_seed':r['seed']=1234567890
    if case=='future':r['utc']=time.time()+1000
    if case=='unknown':r['path']='/etc/passwd'
    with pytest.raises(ValueError):validate_jsonl(encode(rows))

def test_heldout_relabel_rejected(incidents):
    fingerprint=trajectory_hash(incidents[0]);incidents[0]['seed']=5678
    with pytest.raises(ValueError,match='held-out'):validate_jsonl(encode(incidents),{fingerprint})

def test_size_rows_and_confirmation(tmp_path,incidents):
    with pytest.raises(ValueError):validate_jsonl('x'*(MAX_BYTES+1))
    with pytest.raises(ValueError):validate_jsonl(encode(incidents[:32]))
    with pytest.raises(ValueError):Catalog(tmp_path).upload('a@example.com',dict(body(incidents),training_only=False))
    assert not (tmp_path/'datasets').exists()

def test_catalog_immutability_ownership_and_training(tmp_path,incidents):
    c=Catalog(tmp_path);d=c.upload('a@example.com',body(incidents))
    assert c.visible('b@example.com')==[c.public(c.get('expanded-128'))]
    with pytest.raises(PermissionError):c.get(d['id'],'b@example.com')
    assert c.get(d['id'],ADMIN)['owner']=='a@example.com'
    path=c.verify(d['id'],d['sha256']);assert len(validate_jsonl(path.read_text()))==64
    from microservice.probe_target import vector
    assert vector(json.loads(path.read_text().splitlines()[0]),7).shape==(14,)
    with pytest.raises(ValueError,match='already uploaded'):c.upload('a@example.com',body(incidents))
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='hash mismatch'):c.verify(d['id'],d['sha256'])

def test_api_upload_select_download_and_owner(tmp_path,incidents,monkeypatch):
    monkeypatch.setenv('RUNNER_IMAGE',IMAGE);monkeypatch.setenv('CODE_VERSION','test')
    async def check():
        key=b'key';s=Store(tmp_path/'queue')
        async with TestClient(TestServer(create_app(s,tmp_path,key))) as client:
            async def request(method,path,email='a@example.com',value=None,origin=ORIGIN):
                raw=json.dumps(value).encode() if value is not None else b'';stamp=str(time.time())
                headers={'X-Swm-Authenticated-Email':email,'X-Swm-Timestamp':stamp,'Origin':origin,'Content-Type':'application/json','X-Swm-Signature':signature(key,method,path,email,stamp,raw)}
                return await client.request(method,path,data=raw,headers=headers)
            assert (await request('POST','/api/datasets',value=body(incidents),origin='https://evil.example')).status==403
            bad=body(incidents);bad['jsonl']='not jsonl'
            assert (await request('POST','/api/datasets',value=bad)).status==400
            assert not (tmp_path/'datasets').exists()
            response=await request('POST','/api/datasets',value=body(incidents));assert response.status==201
            d=await response.json();assert 'path' not in d
            cfg=await (await request('GET','/api/config')).json();assert d['id'] in [x['id'] for x in cfg['datasets']]
            for email in ('a@example.com',ADMIN):
                result=await request('GET','/api/datasets/'+d['id']+'/download',email=email)
                assert result.status==200;assert len(validate_jsonl(await result.text()))==64
            assert (await request('GET','/api/datasets/'+d['id']+'/download',email='b@example.com')).status==403
            args=dict(ARGS,dataset=d['id'])
            assert (await request('POST','/api/runs',email='b@example.com',value=args)).status==403
            result=await request('POST','/api/runs',value=args);assert result.status==201
            run=await result.json();assert run['dataset_sha256']==d['sha256'];assert run['training_seeds']==[1234]
            stored=s.get(run['id']);command=job_manifest(stored,IMAGE)['spec']['template']['spec']['containers'][0]['command']
            assert command[-2:]==['--dataset-sha256',d['sha256']]
            assert stored['test_seed'] not in stored['training_seeds']
    asyncio.run(check())

def test_retained_test_trajectory_protected(tmp_path,incidents):
    p=tmp_path/'runs'/'old-run';p.mkdir(parents=True);(p/'test.jsonl').write_text(encode(incidents[:1]))
    with pytest.raises(ValueError,match='held-out'):Catalog(tmp_path).upload('a@example.com',body(incidents))


def test_legacy_training_split_and_test_marker(incidents):
    for r in incidents:r.pop('regime');r['split']='development'
    assert len(validate_jsonl(encode(incidents)))==64
    incidents[0]['split']='test'
    with pytest.raises(ValueError,match='evaluation'):validate_jsonl(encode(incidents))
