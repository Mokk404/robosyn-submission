"""Verify the retained policy tree without requiring participant root files."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from _nyx_bundle.assets import stage
ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024), b''):
            h.update(data)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', action='store_true', help='Also verify all 396 local runtime assets')
    args = parser.parse_args()
    errors = []
    tasks = json.loads((ROOT / '_nyx_bundle/TASKS.json').read_text())['tasks']
    for task in tasks:
        for name in ('__init__.py','deploy_policy.py','deploy_policy.yml','eval.sh'):
            if not (ROOT / ('nyx_'+task) / name).is_file():
                errors.append(f'Missing adapter file: {task}/{name}')
    for line in (ROOT / 'SOURCE_SHA256SUMS').read_text().splitlines():
        expected, rel = line.split('  ',1)
        file = ROOT / rel
        if not file.is_file() or sha(file) != expected:
            errors.append(f'Source checksum: {rel}')
    provenance = json.loads((ROOT / 'PACKAGING_PROVENANCE.json').read_text())
    for record in provenance['frozen_files']:
        file = ROOT / record['path']
        if not file.is_file() or sha(file) != record['sha256']:
            errors.append(f'Frozen baseline changed: {record["path"]}')
    for file in ROOT.rglob('*.py'):
        if '.cache' in file.parts or '__pycache__' in file.parts:
            continue
        try: ast.parse(file.read_text(encoding='utf-8-sig'))
        except Exception as exc: errors.append(f'{file}: {exc}')
    asset_result = None
    if args.assets:
        try: asset_result = stage(verify_only=True)
        except Exception as exc: errors.append(str(exc))
    print(json.dumps({'tasks_checked':len(tasks),'frozen_files_checked':len(provenance['frozen_files']),
                      'assets':asset_result,'errors':errors,'gpu_rollout':'not_run'},indent=2))
    return bool(errors)
if __name__ == '__main__':
    sys.exit(main())
