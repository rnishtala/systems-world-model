"""Authenticated experiment API; never accepts commands, images or arbitrary paths."""
import asyncio,fcntl,json,os,re,time
from pathlib import Path
from aiohttp import web
from launcher.auth import verify
from launcher.common import ADMIN,ORIGIN,DATASETS,STEPS,identity
from launcher.store import Store
from launcher.datasets import Catalog

PUBLIC=('id','name','owner','created','status','phase','dataset','seed','test_seed','steps','started','finished','error','version','restored','dataset_name','dataset_sha256','training_seeds','expansion_seed','baseline_run','mode','evaluate_run','training_source','probe_schedule','probe_training_seeds','forecast_source','audit_source','triage','fresh_triage')

def create_app(store,root,key,origin=ORIGIN):
    root=Path(root);catalog=Catalog(root)
    @web.middleware
    async def auth(request,handler):
        if request.path=='/healthz':return web.json_response({'ok':True})
        try:
            email=identity(request.headers);body=await request.read()
            if not verify(key,request.method,request.raw_path,email,request.headers.get('X-Swm-Timestamp'),request.headers.get('X-Swm-Signature',''),body):raise PermissionError('Untrusted ingress')
            if request.method!='GET' and (request.headers.get('Origin')!=origin or request.content_type!='application/json'):raise PermissionError('Same-origin JSON request required')
            request['email']=email
            response=await handler(request)
            response.headers['Cache-Control']='no-store';return response
        except PermissionError as e:raise web.HTTPForbidden(text=str(e))
        except (ValueError,KeyError,RecursionError,json.JSONDecodeError) as e:raise web.HTTPBadRequest(text=str(e))
    app=web.Application(middlewares=[auth],client_max_size=4*1024*1024)
    def authorized(request):
        uid=request.match_info['uid']
        if not re.fullmatch('[a-f0-9]{12}',uid):raise web.HTTPNotFound()
        try:run=store.get(uid)
        except KeyError:raise web.HTTPNotFound()
        if request['email'] not in (run['owner'],ADMIN):raise PermissionError('Only the owner or administrator can access this run')
        return run
    def public(run):
        result={k:run[k] for k in PUBLIC if k in run};path=root/'runs'/run['id']
        progress=path/'progress.json'
        try:
            if run['status']=='running':result['phase']=json.loads(progress.read_text())['phase']
        except (OSError,ValueError,KeyError):pass
        test=path/'test.jsonl'
        if test.exists():result['collected']=sum(1 for _ in test.open())
        log=path/'run.log'
        if log.exists():
            with log.open('rb') as f:f.seek(max(0,log.stat().st_size-12000));tail=f.read().decode(errors='replace')
            steps=re.findall(r'Agent Step ([\d_]+)',tail)
            if steps:result['training_step']=int(steps[-1].replace('_',''))
        protocol=path/'protocol.json'
        if protocol.exists():
            try:
                info=json.loads(protocol.read_text())
                result['protocol']={k:info[k] for k in ('probe_schedule','comparison_scope','training_incidents','training_sha256','training_collection_seeds') if k in info}
            except (OSError,ValueError):pass
        summary=path/'evaluation/summary.json'
        if summary.exists():
            try:result['summary']=json.loads(summary.read_text())
            except ValueError:pass
        diagnostic=path/'forecast-diagnostic/summary.json'
        if diagnostic.exists():
            try:result['diagnostic_summary']=json.loads(diagnostic.read_text())
            except (ValueError,OSError):pass
        forecast=path/'forecast-evaluation/summary.json'
        if forecast.exists():
            try:result['forecast_summary']=json.loads(forecast.read_text())
            except (OSError,ValueError):pass
        return result
    async def config(request):return web.json_response(dict(email=request['email'],admin=request['email']==ADMIN,version=os.environ.get('CODE_VERSION','development'),datasets=catalog.visible(request['email']),steps=list(STEPS)))
    async def listing(request):return web.json_response([public(r) for r in reversed(store.all()) if request['email'] in (r['owner'],ADMIN)])
    async def create(request):
        body=await request.json()
        if not isinstance(body,dict):raise ValueError('Expected an experiment object')
        dataset=catalog.get(body.get('dataset'),request['email'])
        run=store.create(request['email'],body,os.environ['RUNNER_IMAGE'],os.environ['CODE_VERSION'],dataset);return web.json_response(public(run),status=201)
    async def upload(request):
        result=catalog.upload(request['email'],await request.json())
        return web.json_response(result,status=201)
    async def download_dataset(request):
        d=catalog.get(request.match_info['dataset_id'],request['email'])
        path=catalog.verify(d['id'],d['sha256'])
        return web.Response(body=path.read_bytes(),content_type='application/x-ndjson',headers={'Content-Disposition':f'attachment; filename="{d["id"]}.jsonl"'})
    async def detail(request):return web.json_response(public(authorized(request)))
    async def cancel(request):
        run=authorized(request)
        if run['status']=='queued':run.update(status='cancelled',phase='cancelled',finished=time.time())
        elif run['status'] not in ('completed','failed','cancelled'):run['cancel']=True
        store.save(run);return web.json_response(public(run))
    async def logs(request):
        run=authorized(request);path=root/'runs'/run['id']/'run.log'
        if not path.exists():return web.Response(text='Waiting for worker logs…')
        with path.open('rb') as f:f.seek(max(0,path.stat().st_size-32000));text=f.read().decode(errors='replace')
        return web.Response(text=text)
    async def report(request):
        run=authorized(request);data=public(run)
        return web.Response(text=json.dumps(data,indent=2),content_type='application/json',headers={'Content-Disposition':f'attachment; filename="experiment-{run["id"]}.json"'})
    app.router.add_post('/api/datasets',upload);app.router.add_get('/api/datasets/{dataset_id}/download',download_dataset);app.router.add_get('/api/config',config);app.router.add_get('/api/runs',listing);app.router.add_post('/api/runs',create)
    app.router.add_get('/api/runs/{uid}',detail);app.router.add_post('/api/runs/{uid}/cancel',cancel);app.router.add_get('/api/runs/{uid}/logs',logs);app.router.add_get('/api/runs/{uid}/report',report)
    return app

async def main():
    from launcher.kube import Kube
    from launcher.controller import Controller
    root=Path(os.environ.get('RESULTS_ROOT','/results/experiments'));root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    store=Store(root/'queue.sqlite');kube=Kube();controller=Controller(store,kube,root)
    key=Path('/auth/key').read_bytes();app=create_app(store,root,key)
    runner=web.AppRunner(app);await runner.setup();await web.TCPSite(runner,'0.0.0.0',8090).start()
    try:
        while True:
            try:await controller.tick()
            except Exception as e:print('CONTROLLER_ERROR',str(e),flush=True)
            await asyncio.sleep(5)
    finally:await kube.close();await runner.cleanup();lock.close()
if __name__=='__main__':asyncio.run(main())
