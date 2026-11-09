"""Generate namespace-bound microservice resources; never target other estates."""
import json
from pathlib import Path
NS='systems-world-model-lab';items=[]
def obj(kind,name,**extra):
 d={'apiVersion':'v1','kind':kind,'metadata':{'name':name,'namespace':NS},**extra};items.append(d);return d
security={'runAsNonRoot':True,'runAsUser':1000,'runAsGroup':1000,'fsGroup':1000,'seccompProfile':{'type':'RuntimeDefault'}}
container_security={'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}}
obj('ResourceQuota','lab-budget',spec={'hard':{'requests.cpu':'4','limits.cpu':'4','requests.memory':'6Gi','limits.memory':'6Gi','pods':'8','persistentvolumeclaims':'1','requests.storage':'10Gi'}})
for role,cpu,mem in [('checkout','300m','128Mi'),('payment','600m','256Mi'),('ledger','400m','256Mi')]:
 labels={'swm-component':role}
 env=[{'name':'SERVICE_ROLE','value':role},{'name':'PYTHONDONTWRITEBYTECODE','value':'1'},{'name':'PYTHONPATH','value':'/app/deps'}]
 if role!='ledger':env.append({'name':'DOWNSTREAM','value':'http://swm-'+('payment' if role=='checkout' else 'ledger')+':8080'})
 spec={'automountServiceAccountToken':False,'securityContext':security,'nodeSelector':{'kubernetes.io/hostname':'amd-halo'},
 'initContainers':[{'name':'unpack','image':'python:3.11-slim','command':['sh','-ec','tar xzf /source/app.tar.gz -C /app'],'resources':{'requests':{'cpu':'50m','memory':'64Mi'},'limits':{'cpu':cpu,'memory':mem}},'securityContext':container_security,'volumeMounts':[{'name':'source','mountPath':'/source','readOnly':True},{'name':'app','mountPath':'/app'}]}],
 'containers':[{'name':role,'image':'python:3.11-slim','command':['python','/app/microservice/service.py'],'env':env,'securityContext':container_security,'resources':{'requests':{'cpu':'50m','memory':'64Mi'},'limits':{'cpu':cpu,'memory':mem}},'readinessProbe':{'httpGet':{'path':'/health','port':8080},'periodSeconds':3},'volumeMounts':[{'name':'app','mountPath':'/app','readOnly':True},{'name':'tmp','mountPath':'/tmp'}]}],
 'volumes':[{'name':'source','configMap':{'name':'microservice-source-v1'}},{'name':'app','emptyDir':{'sizeLimit':'32Mi'}},{'name':'tmp','emptyDir':{'sizeLimit':'16Mi'}}]}
 d=obj('Deployment','swm-'+role,spec={'replicas':1,'strategy':{'type':'Recreate'},'selector':{'matchLabels':labels},'template':{'metadata':{'labels':labels},'spec':spec}});d['apiVersion']='apps/v1'
 obj('Service','swm-'+role,spec={'selector':labels,'ports':[{'name':'http','port':8080},{'name':'control','port':9090}]})
labels={'swm-component':'postgres'}
d=obj('Deployment','swm-postgres',spec={'replicas':1,'strategy':{'type':'Recreate'},'selector':{'matchLabels':labels},'template':{'metadata':{'labels':labels},'spec':{'automountServiceAccountToken':False,'securityContext':{**security,'runAsUser':999,'runAsGroup':999,'fsGroup':999},'nodeSelector':{'kubernetes.io/hostname':'amd-halo'},'containers':[{'name':'postgres','image':'postgres:16-bookworm','args':['-c','shared_buffers=32MB','-c','max_connections=32'],'env':[{'name':'POSTGRES_HOST_AUTH_METHOD','value':'trust'},{'name':'PGDATA','value':'/var/lib/postgresql/data/pgdata'}],'resources':{'requests':{'cpu':'50m','memory':'128Mi'},'limits':{'cpu':'300m','memory':'384Mi'}},'securityContext':container_security,'readinessProbe':{'exec':{'command':['pg_isready','-U','postgres']},'periodSeconds':3},'volumeMounts':[{'name':'data','mountPath':'/var/lib/postgresql/data'},{'name':'run','mountPath':'/var/run/postgresql'},{'name':'tmp','mountPath':'/tmp'}]}],'volumes':[{'name':'data','emptyDir':{'sizeLimit':'512Mi'}},{'name':'run','emptyDir':{'sizeLimit':'16Mi'}},{'name':'tmp','emptyDir':{'sizeLimit':'16Mi'}}]}}});d['apiVersion']='apps/v1'
obj('Service','swm-postgres',spec={'selector':labels,'ports':[{'name':'postgres','port':5432}]})
def peer(key,value):return {'podSelector':{'matchLabels':{key:value}}}
def net(name,selector,ingress=None,egress=None):
 spec={'podSelector':{'matchLabels':selector},'policyTypes':['Ingress','Egress']}
 if ingress is not None:spec['ingress']=ingress
 if egress is not None:spec['egress']=egress
 d=obj('NetworkPolicy',name,spec=spec);d['apiVersion']='networking.k8s.io/v1'
dns={'to':[{'namespaceSelector':{'matchLabels':{'kubernetes.io/metadata.name':'kube-system'}},'podSelector':{'matchLabels':{'k8s-app':'kube-dns'}}}],'ports':[{'protocol':'UDP','port':53},{'protocol':'TCP','port':53}]}
for role in ('checkout','payment','ledger'):
 callers=[peer('swm-role','runner'),peer('swm-role','agent')]
 if role!='checkout':callers.append(peer('swm-component','checkout' if role=='payment' else 'payment'))
 ing=[{'from':callers,'ports':[{'port':8080}]},{'from':[peer('swm-role','runner')],'ports':[{'port':9090}]}]
 target='payment' if role=='checkout' else 'ledger' if role=='payment' else 'postgres'
 net('swm-'+role,{'swm-component':role},ing,[dns,{'to':[peer('swm-component',target)],'ports':[{'port':5432 if role=='ledger' else 8080}]}])
net('swm-postgres',{'swm-component':'postgres'},[{'from':[peer('swm-component','ledger')],'ports':[{'port':5432}]}],[])
for role in ('runner','agent'):
 net('swm-'+role,{'swm-role':role},[],[dns,{'to':[peer('swm-component',s) for s in ('checkout','payment','ledger')],'ports':[{'port':8080}]+([{'port':9090}] if role=='runner' else [])}])
Path('deploy/microservice-lab.json').write_text(json.dumps({'apiVersion':'v1','kind':'List','items':items},indent=2)+'\n')
