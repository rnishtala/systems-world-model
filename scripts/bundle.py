"""Bundle this repository and an exact upstream checkout for ConfigMap delivery."""
import argparse
import subprocess
import tarfile
from pathlib import Path

REVISION = 'e01491fad6434b2245a3b8ca201dd7faedcc458c'
p = argparse.ArgumentParser()
p.add_argument('upstream', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
actual = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
if actual != REVISION:
    raise SystemExit('Unexpected Dreamer revision')
if subprocess.check_output(['git', '-C', str(a.upstream), 'status', '--porcelain'], text=True):
    raise SystemExit('Upstream checkout must be clean')
with tarfile.open(a.output, 'w:gz') as archive:
    def clean(info):
        return None if '__pycache__' in info.name or '.pytest_cache' in info.name else info
    for name in ['experiment', 'tests', 'requirements-cpu.txt']:
        archive.add(name, arcname=name, filter=clean)
    for name in ['dreamerv3', 'embodied', 'LICENSE']:
        archive.add(a.upstream / name, arcname='upstream/' + name, filter=clean)
print(a.output)
