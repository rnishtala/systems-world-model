"""Generate namespace-scoped launcher resources from immutable build output."""
import json,re,sys
from pathlib import Path
from launcher.common import NAMESPACE

def resources(images):
    for kind in ('control','runner'):
        if not re.fullmatch('ghcr.io/rnishtala/systems-world-model-'+kind+r'@sha256:[a-f0-9]{64}',images[kind]):raise ValueError('Expected immutable image digest')
    def obj(api,kind,name,**kw):return dict(apiVersion=api,kind=kind,metadata=dict(name=name,namespace=NAMESPACE),**kw)
    label={'app':'swm-experiments'};rbac='rbac.authorization.k8s.io/v1'
    security=dict(allowPrivilegeEscalation=False,readOnlyRootFilesystem=True,capabilities=dict(drop=['ALL']))
    rules=[dict(apiGroups=['batch'],resources=['jobs'],verbs=['get','list','create','delete']),dict(apiGroups=[''],resources=['pods'],verbs=['list']),dict(apiGroups=[''],resources=['resourcequotas'],resourceNames=['lab-budget'],verbs=['get']),dict(apiGroups=['apps'],resources=['deployments'],resourceNames=['swm-inference'],verbs=['get']),dict(apiGroups=['apps'],resources=['deployments/scale'],resourceNames=['swm-inference'],verbs=['get','patch'])]
    items=[obj('v1','ServiceAccount','swm-experiments'),obj(rbac,'Role','swm-experiments',rules=rules),obj(rbac,'RoleBinding','swm-experiments',roleRef=dict(apiGroup='rbac.authorization.k8s.io',kind='Role',name='swm-experiments'),subjects=[dict(kind='ServiceAccount',name='swm-experiments',namespace=NAMESPACE)])]
    container=dict(name='controller',image=images['control'],env=[dict(name='RUNNER_IMAGE',value=images['runner']),dict(name='CODE_VERSION',value=images['version'])],resources=dict(requests=dict(cpu='10m',memory='64Mi'),limits=dict(cpu='25m',memory='128Mi')),securityContext=security,volumeMounts=[dict(name='results',mountPath='/results'),dict(name='auth',mountPath='/auth',readOnly=True)],readinessProbe=dict(httpGet=dict(path='/healthz',port=8090),initialDelaySeconds=3))
    items.append(obj('apps/v1','Deployment','swm-experiments',spec=dict(replicas=1,strategy=dict(type='Recreate'),selector=dict(matchLabels=label),template=dict(metadata=dict(labels=label),spec=dict(serviceAccountName='swm-experiments',imagePullSecrets=[dict(name='swm-experiment-pull')],securityContext=dict(runAsNonRoot=True,runAsUser=1000,runAsGroup=1000,fsGroup=1000,seccompProfile=dict(type='RuntimeDefault')),containers=[container],volumes=[dict(name='results',persistentVolumeClaim=dict(claimName='results')),dict(name='auth',secret=dict(secretName='swm-experiment-auth'))])))))
    items.append(obj('v1','Service','swm-experiments',spec=dict(selector=label,ports=[dict(port=8090,targetPort=8090)])))
    peer=dict(podSelector=dict(matchLabels={'app':'swm-headlamp-public'}))
    dns=dict(namespaceSelector=dict(matchLabels={'kubernetes.io/metadata.name':'kube-system'}),podSelector=dict(matchLabels={'k8s-app':'kube-dns'}))
    items.append(obj('networking.k8s.io/v1','NetworkPolicy','swm-experiments',spec=dict(podSelector=dict(matchLabels=label),policyTypes=['Ingress','Egress'],ingress=[{'from':[peer],'ports':[dict(port=8090,protocol='TCP')]}],egress=[{'to':[dict(ipBlock=dict(cidr=ip)) for ip in ['10.43.0.1/32','100.98.114.55/32']],'ports':[dict(port=p,protocol='TCP') for p in [443,6443]]},{'to':[dns],'ports':[dict(port=53,protocol=p) for p in ['UDP','TCP']]}])))
    items.append(obj('networking.k8s.io/v1','NetworkPolicy','swm-headlamp-experiments',spec=dict(podSelector=peer['podSelector'],policyTypes=['Egress'],egress=[{'to':[dict(podSelector=dict(matchLabels=label))],'ports':[dict(port=8090,protocol='TCP')]}])))
    items.append(obj('v1','ConfigMap','swm-headlamp-experiments',data={'main.js':Path(__file__).with_name('headlamp-plugin.js').read_text(),'package.json':json.dumps({'name':'swm-experiments','version':'0.1.1','main':'main.js','devDependencies':{'@kinvolk/headlamp-plugin':'0.13.1'}})}))
    return dict(apiVersion='v1',kind='List',items=items)

if __name__=='__main__':print(json.dumps(resources(json.loads(Path(sys.argv[1]).read_text()))))
