import argparse,json,urllib.request,urllib.error
from pathlib import Path
p=argparse.ArgumentParser(description='Check operator access through Headlamp; mutations are dry-run only.');p.add_argument('--token-file',required=True,type=Path);a=p.parse_args()
R=Path(__file__).resolve().parents[1]; token=a.token_file.read_text().strip();base='http://amd-halo.tailf2f933.ts.net:8083/clusters/systems-world-model-lab'
checks=[]
def call(path,method='GET',data=None,auth=True,ctype='application/json'):
 h={'Content-Type':ctype}
 if auth:h['Authorization']='Bearer '+token
 req=urllib.request.Request(base+path,headers=h,method=method,data=None if data is None else json.dumps(data).encode())
 try:
  with urllib.request.urlopen(req,timeout=20) as r:return r.status,json.load(r)
 except urllib.error.HTTPError as e:return e.code,{}
def check(name,path,want,**kw):
 code,data=call(path,**kw);checks.append({'check':name,'expected':want,'actual':code});assert code==want,(name,code);return data
ns='systems-world-model-lab'
check('authenticated lab pods',f'/api/v1/namespaces/{ns}/pods',200)
check('other namespace blocked','/api/v1/namespaces/default/pods',403)
check('cluster nodes blocked','/api/v1/nodes',403)
check('secrets blocked',f'/api/v1/namespaces/{ns}/secrets',403)
check('anonymous blocked',f'/api/v1/namespaces/{ns}/pods',401,auth=False)
for name,verb,group,resource,namespace,resname,allowed in [
 ('view logs','get','','pods/log',ns,'',True),('create jobs','create','batch','jobs',ns,'',True),
 ('patch lab service workload','patch','apps','deployments',ns,'swm-checkout',True),
 ('scale inference','patch','apps','deployments/scale',ns,'swm-inference',True),
 ('edit headlamp blocked','patch','apps','deployments',ns,'swm-headlamp',False),
 ('create deployments blocked','create','apps','deployments',ns,'',False),
 ('change quota blocked','patch','','resourcequotas',ns,'lab-budget',False),
 ('change networkpolicy blocked','patch','networking.k8s.io','networkpolicies',ns,'swm-headlamp',False),
 ('delete PVC blocked','delete','','persistentvolumeclaims',ns,'results',False),
 ('grant permissions blocked','create','rbac.authorization.k8s.io','rolebindings',ns,'',False),
 ('mint tokens blocked','create','','serviceaccounts/token',ns,'',False),
 ('other namespace writes blocked','create','batch','jobs','default','',False),
 ('exec blocked','create','','pods/exec',ns,'',False),
 ('impersonate blocked','impersonate','','users','','',False)]:
 attrs=dict(verb=verb,group=group,resource=resource,namespace=namespace)
 if resname:attrs['name']=resname
 code,d=call('/apis/authorization.k8s.io/v1/selfsubjectaccessreviews',method='POST',data={'apiVersion':'authorization.k8s.io/v1','kind':'SelfSubjectAccessReview','spec':{'resourceAttributes':attrs}})
 actual=d.get('status',{}).get('allowed');assert code==201 and actual is allowed,(name,code,d)
 checks.append({'check':name,'expected':allowed,'actual':actual})
check('deployment patch server dry-run',f'/apis/apps/v1/namespaces/{ns}/deployments/swm-checkout?dryRun=All',200,method='PATCH',ctype='application/merge-patch+json',data={'metadata':{'annotations':{'swm-ui-verification':'dry-run-only'}}})
job={'apiVersion':'batch/v1','kind':'Job','metadata':{'name':'swm-ui-access-check'},'spec':{'ttlSecondsAfterFinished':86400,'activeDeadlineSeconds':60,'template':{'spec':{'automountServiceAccountToken':False,'restartPolicy':'Never','securityContext':{'runAsNonRoot':True,'runAsUser':1000,'seccompProfile':{'type':'RuntimeDefault'}},'containers':[{'name':'check','image':'busybox:1.37','command':['true'],'securityContext':{'allowPrivilegeEscalation':False,'capabilities':{'drop':['ALL']}},'resources':{'limits':{'cpu':'10m','memory':'16Mi'}}}]}}}}
check('job creation server dry-run',f'/apis/batch/v1/namespaces/{ns}/jobs?dryRun=All',201,method='POST',data=job)
(R/'reports/evidence/headlamp-access').mkdir(exist_ok=True)
(R/'reports/evidence/headlamp-access/checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({'passed':len(checks),'checks':checks},indent=2))
