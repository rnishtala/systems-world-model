"""Stream an allowlisted evidence archive from the read-only results collector.

Usage: python export_confirmation.py /results/confirmation-v4 > evidence.tar.gz
Checkpoints and replay buffers stay in the lab PVC; no arbitrary files are exported.
"""
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
paths = []
for seed in (10, 11, 12):
    paths.extend(root / name for name in (
        f'results-{seed}.json', f'particle-results-{seed}.json',
        f'train-{seed}.log', f'train-{seed}/config.yaml',
        f'train-{seed}/metrics.jsonl',
    ))
    scores = root / f'train-{seed}/scores.jsonl'
    if scores.exists():
        paths.append(scores)
for path in paths:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Missing or unexpected evidence file: {path}')
for seed in (10, 11, 12):
    for name in (f'results-{seed}.json', f'particle-results-{seed}.json'):
        result = json.loads((root / name).read_text())
        if result['training_seed'] != seed:
            raise ValueError(f'Seed mismatch: {name}')
if (root / 'demo.json').exists():
    paths.append(root / 'demo.json')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for path in paths:
        archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
