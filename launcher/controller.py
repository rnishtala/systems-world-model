"""Reconcile the durable queue, including interrupted launches and restoration."""
import copy,json,shutil,time
from pathlib import Path
from launcher.common import ACTIVE,job_manifest,dataset_check,run_budget
from launcher.kube import KubeError

def cpu(value):return float(value[:-1])/1000 if str(value).endswith('m') else float(value)

class Controller:
    def __init__(self,store,kube,root):self.store=store;self.kube=kube;self.root=Path(root)
    async def tick(self):
        rows=self.store.all();active=[r for r in rows if r['status'] in ACTIVE]
        if len(active)>1:raise RuntimeError('Multiple active runs; refusing to mutate the lab')
        if active:run=active[0]
        else:
            pending=[r for r in rows if r['status']=='queued']
            if not pending:return
            run=pending[0]
        try:await self.advance(run)
        except Exception as e:
            run['error']=str(e)[:1000];run['phase']='Waiting for controller recovery'
            self.store.save(run)
    async def advance(self,run):
        uid=run['id'];jobs=await self.kube.jobs();job=next((j for j in jobs if j['metadata']['name']=='swm-run-'+uid),None)
        if job is not None and job['metadata'].get('labels',{}).get('swm-experiment')!=uid:raise RuntimeError('Job name belongs to another workflow')
        if run['status']=='queued':
            if any(not j.get('status',{}).get('succeeded') and not j.get('status',{}).get('failed') for j in jobs):
                run['phase']='Waiting for the lab to be free';self.store.save(run);return
            try:dataset_check(run['dataset'],run.get('dataset_sha256'))
            except Exception as e:
                run.update(status='failed',phase='Preflight failed',error=str(e));self.store.save(run);return
            if shutil.disk_usage(self.root).free<1024**3:raise RuntimeError('Less than 1 GiB of free results storage')
            deployment=await self.kube.deployment()
            run.update(status='preparing',phase='Reserving lab resources',restore_spec=deployment['spec'],started=time.time())
            self.store.save(run)  # Durable restore intent precedes every scale operation.
        if run['status']=='preparing':
            if run['cancel']:
                run.update(status='restoring',outcome='cancelled',release_after=time.time());self.store.save(run)
            else:
                deployment=await self.kube.deployment()
                # Do not silently overwrite unrelated deployment edits.
                original=copy.deepcopy(run['restore_spec']);current=copy.deepcopy(deployment['spec'])
                original.pop('replicas',None);current.pop('replicas',None)
                if current!=original:raise RuntimeError('Inference deployment changed; operator review required')
                if deployment['spec'].get('replicas',1)!=0:await self.kube.scale(0)
                if job is None:
                    quota=await self.kube.quota()
                    if cpu(quota['status']['used']['limits.cpu'])>2:return
                    try:job=await self.kube.create(job_manifest(run,run['image']))
                    except KubeError as e:
                        if e.status==409:return  # Response may have been lost; reconcile, never recreate.
                        if e.status in (400,403,422):
                            run.update(status='restoring',outcome='failed',error=str(e),release_after=time.time());self.store.save(run);return
                        raise
                run.update(status='running',phase='Starting worker',error=None);self.store.save(run)
        if run['status']=='running':
            progress=self.root/'runs'/uid/'progress.json'
            if progress.exists():
                try:run['phase']=json.loads(progress.read_text())['phase']
                except (ValueError,KeyError):pass
            if run['cancel']:
                run.update(status='restoring',outcome='cancelled',release_after=time.time()+45)
            elif job is None:
                complete=(self.root/'runs'/uid/('temporal-pilot' if run.get('temporal_pilot') else 'forecast-diagnostic' if run.get('audit_source') or run.get('fresh_triage') else 'forecast-evaluation' if run.get('mode')=='forecast' else 'evaluation')/'summary.json').exists()
                run.update(status='restoring',outcome='completed' if complete else 'failed',error=None if complete else 'Worker disappeared before completion',release_after=time.time()+45)
            elif job.get('status',{}).get('succeeded'):
                run.update(status='restoring',outcome='completed',release_after=time.time())
            elif job.get('status',{}).get('failed') or time.time()-run['started']>run_budget(run)+300:
                run.update(status='restoring',outcome='failed',error='Worker failed or exceeded its time budget; inspect the run log',release_after=time.time()+45)
            self.store.save(run)
        if run['status']=='restoring':
            run['phase']='Restoring lab';self.store.save(run)
            await self.kube.delete(uid)
            if await self.kube.pods(uid):return
            deployment=await self.kube.deployment();expected=run['restore_spec'].get('replicas',1)
            original=copy.deepcopy(run['restore_spec']);current=copy.deepcopy(deployment['spec'])
            original.pop('replicas',None);current.pop('replicas',None)
            if current!=original:raise RuntimeError('Inference deployment changed; refusing to overwrite it during restoration')
            if deployment['spec'].get('replicas',1)!=expected:await self.kube.scale(expected);return
            if deployment.get('status',{}).get('readyReplicas',0)<expected:return
            if time.time()<run.get('release_after',0):return
            run.update(status=run['outcome'],phase=run['outcome'],finished=time.time(),restored=True)
            self.store.save(run)
