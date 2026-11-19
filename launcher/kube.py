"""Minimal Kubernetes client restricted to this namespace's resource paths."""
import os,ssl
from pathlib import Path
from urllib.parse import quote
import aiohttp
from launcher.common import NAMESPACE

class KubeError(RuntimeError):
    def __init__(self,status,message):super().__init__(message);self.status=status

class Kube:
    def __init__(self):
        self.root=Path('/var/run/secrets/kubernetes.io/serviceaccount')
        self.ssl=ssl.create_default_context(cafile=str(self.root/'ca.crt'))
        self.url='https://'+os.environ['KUBERNETES_SERVICE_HOST']+':'+os.environ.get('KUBERNETES_SERVICE_PORT','443')
        self.session=aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20))
    async def call(self,method,path,body=None):
        prefix=('/api/v1/namespaces/'+NAMESPACE+'/', '/apis/apps/v1/namespaces/'+NAMESPACE+'/', '/apis/batch/v1/namespaces/'+NAMESPACE+'/')
        if not path.startswith(prefix):raise ValueError('Namespace boundary')
        headers={'Authorization':'Bearer '+(self.root/'token').read_text().strip()}
        if method=='PATCH':headers['Content-Type']='application/merge-patch+json'
        async with self.session.request(method,self.url+path,json=body,headers=headers,ssl=self.ssl) as response:
            text=await response.text()
            if response.status>=400:raise KubeError(response.status,text[:1500])
            import json
            return json.loads(text)
    async def jobs(self):return (await self.call('GET',f'/apis/batch/v1/namespaces/{NAMESPACE}/jobs'))['items']
    async def pods(self,uid):return (await self.call('GET',f'/api/v1/namespaces/{NAMESPACE}/pods?labelSelector=swm-experiment%3D'+quote(uid)))['items']
    async def deployment(self):return await self.call('GET',f'/apis/apps/v1/namespaces/{NAMESPACE}/deployments/swm-inference')
    async def scale(self,n):return await self.call('PATCH',f'/apis/apps/v1/namespaces/{NAMESPACE}/deployments/swm-inference/scale',{'spec':{'replicas':n}})
    async def create(self,manifest):return await self.call('POST',f'/apis/batch/v1/namespaces/{NAMESPACE}/jobs',manifest)
    async def delete(self,uid):
        try:await self.call('DELETE',f'/apis/batch/v1/namespaces/{NAMESPACE}/jobs/swm-run-'+uid,{'propagationPolicy':'Foreground'})
        except KubeError as e:
            if e.status!=404:raise
    async def quota(self):return await self.call('GET',f'/api/v1/namespaces/{NAMESPACE}/resourcequotas/lab-budget')
    async def close(self):await self.session.close()
