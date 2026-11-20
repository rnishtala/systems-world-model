import hashlib,json,re,time,uuid
from pathlib import Path

NAMESPACE='systems-world-model-lab'
ADMIN='099.raj@gmail.com'
ORIGIN='https://noninstructive-unprevalently-minerva.ngrok-free.dev'
DATASETS={'expanded-128':dict(name='Expanded service incidents',incidents=128,path='/results/probe-retrain-v1/collection/train.jsonl',sha256='10e6cd7a2d6ae5c3e19373114ea763fc759e4a89a9bde7711a2396d0b857e316')}
STEPS=(1000,10000,25000,50000,100000)
ACTIVE=('preparing','running','restoring')
TERMINAL=('completed','failed','cancelled')

def identity(headers):
    # Only the loopback ingress proxy may reach this service (NetworkPolicy).
    values=headers.getall('X-Swm-Authenticated-Email',[]) if hasattr(headers,'getall') else [headers.get('X-Swm-Authenticated-Email','')]
    if len(values)!=1 or not re.fullmatch(r'[A-Za-z0-9.!#$%&\x27*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',values[0]):
        raise PermissionError('Verified Google identity required')
    return values[0].lower()

def validate_request(body,datasets=None):
    if not isinstance(body,dict) or set(body)!={'name','dataset','seed','steps'}:raise ValueError('Expected name, dataset, seed and steps')
    if not isinstance(body['name'],str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}',body['name']):raise ValueError('Use 1–64 letters, numbers, spaces, dots, underscores or hyphens for the name')
    if not isinstance(body['dataset'],str) or body['dataset'] not in (DATASETS if datasets is None else datasets):raise ValueError('Choose an approved dataset')
    if type(body['seed']) is not int or not 1<=body['seed']<=1000000000:raise ValueError('Seed must be an integer from 1 to 1,000,000,000')
    if type(body['steps']) is not int or body['steps'] not in STEPS:raise ValueError('Choose an approved step budget')
    return dict(body)

def dataset_check(name,expected=None):
    from launcher.datasets import Catalog
    return Catalog().verify(name,expected)

def job_manifest(run,image):
    if not re.fullmatch(r'ghcr.io/rnishtala/systems-world-model-runner@sha256:[a-f0-9]{64}',image):raise ValueError('Runner must be pinned by digest')
    uid=run['id'];label={'swm-role':'runner','swm-experiment':uid,'app.kubernetes.io/managed-by':'swm-experiments'}
    command=['python','-m','launcher.pipeline','--run',uid,'--dataset',run['dataset'],'--seed',str(run['seed']),'--steps',str(run['steps']),'--test-seed',str(run['test_seed'])]
    if run.get('dataset_sha256'):command+=['--dataset-sha256',run['dataset_sha256']]
    if run.get('expansion_seed') is not None:
        expansion_args=validate_expansion(run['expansion_seed'],run.get('baseline_run'),run['test_seed'])
        command+=expansion_args
    return dict(apiVersion='batch/v1',kind='Job',metadata=dict(name='swm-run-'+uid,namespace=NAMESPACE,labels=label),spec=dict(backoffLimit=0,activeDeadlineSeconds=3600,ttlSecondsAfterFinished=86400,template=dict(metadata=dict(labels=label),spec=dict(restartPolicy='Never',automountServiceAccountToken=False,imagePullSecrets=[dict(name='swm-experiment-pull')],terminationGracePeriodSeconds=30,securityContext=dict(runAsNonRoot=True,runAsUser=1000,runAsGroup=1000,fsGroup=1000,seccompProfile=dict(type='RuntimeDefault')),containers=[dict(name='experiment',image=image,command=command,env=[dict(name='POD_NAMESPACE',valueFrom=dict(fieldRef=dict(fieldPath='metadata.namespace')))],resources=dict(requests=dict(cpu='1',memory='3Gi'),limits=dict(cpu='2',memory='4Gi')),securityContext=dict(allowPrivilegeEscalation=False,readOnlyRootFilesystem=True,capabilities=dict(drop=['ALL'])),volumeMounts=[dict(name='results',mountPath='/results'),dict(name='tmp',mountPath='/tmp')])],volumes=[dict(name='results',persistentVolumeClaim=dict(claimName='results')),dict(name='tmp',emptyDir=dict(sizeLimit='512Mi'))]))))


def validate_expansion(seed,baseline,test_seed):
    if type(seed) is not int or not 1<=seed<1000000000 or seed==test_seed:
        raise ValueError('Expansion requires a distinct training collection seed')
    if not isinstance(baseline,str) or not re.fullmatch('[a-f0-9]{12}',baseline):
        raise ValueError('Expansion requires a baseline run ID')
    return ['--expand-training-seed',str(seed),'--baseline-run',baseline]
