import ast
from pathlib import Path
from unittest.mock import patch
import numpy as np
from microservice.supervision import telemetry,TARGET
from microservice import live


def test_targets_never_enter_encoder_inputs():
    obs={'vector':np.ones(14),TARGET:np.ones(3)}
    assert TARGET not in telemetry(obs) and TARGET in obs


def test_collection_dns_is_namespace_scoped():
    with patch.object(live.urllib.request,'urlopen',side_effect=RuntimeError('captured')) as call:
        try:live.request('ledger','/health')
        except RuntimeError:pass
        assert call.call_args.args[0].full_url=='http://swm-ledger.systems-world-model-lab.svc.cluster.local:8080/health'


def test_all_local_imports_exist():
    root=Path(__file__).parents[1]
    for folder in ('microservice','scripts','tests'):
        for path in (root/folder).glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                names=[node.module] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names] if isinstance(node,ast.Import) else []
                for name in names:
                    if name and name.split('.')[0] in ('microservice','scripts','tests','experiment','benchmark'):
                        candidate=root.joinpath(*name.split('.'))
                        assert candidate.is_dir() or candidate.with_suffix('.py').exists(),(path,name)
