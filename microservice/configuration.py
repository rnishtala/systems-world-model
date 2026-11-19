"""Pinned upstream size1m configuration."""
import os,sys
from pathlib import Path
sys.path.insert(0,os.environ.get('DREAMER_SOURCE','upstream'))
import elements
from dreamerv3 import main as upstream
import ruamel.yaml as yaml

def configuration(seed, logdir, steps):
    configs = yaml.YAML(typ='safe').load(Path(upstream.__file__).with_name('configs.yaml').read_text())
    c = elements.Config(configs['defaults']).update(configs['size1m'])
    return c.update(dict(seed=seed, logdir=str(logdir), task='triage_queue', batch_size=8,
        batch_length=16, report_length=16, **{
        'jax.platform':'cpu', 'jax.compute_dtype':'float32', 'jax.prealloc':False,
        'run.steps':steps, 'run.envs':1, 'run.train_ratio':16., 'run.log_every':30,
        'run.report_every':300, 'run.save_every':30, 'run.debug':True,
        'run.usage.nvsmi':False, 'replay.size':10000,
        'logger.outputs':['jsonl'], 'agent.report':False}))
