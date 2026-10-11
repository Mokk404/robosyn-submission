"""Package checks; no model loads or simulator startup.

Use --write-integrity only after intentional package edits to rebuild the
top-level checksum manifest. Normal verification remains read-only.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
MANIFEST_NAME='SHA256SUMS_UNIFIED'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def manifest_files():
    for p in sorted(ROOT.rglob('*'), key=lambda item:item.relative_to(ROOT).as_posix()):
        if not p.is_file() or p.name == MANIFEST_NAME:
            continue
        if p.suffix.lower() in {'.mp4','.webm','.mov','.m4v'}:
            continue
        if '_history' in p.parts or any(part.startswith('history_prepackage_') for part in p.parts):
            continue
        yield p

def write_integrity_manifest():
    lines=[f'{digest(p)}  {p.relative_to(ROOT).as_posix()}' for p in manifest_files()]
    (ROOT/MANIFEST_NAME).write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    return len(lines)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--integrity',action='store_true')
    parser.add_argument('--write-integrity',action='store_true')
    args=parser.parse_args()
    written=write_integrity_manifest() if args.write_integrity else 0
    mapping=json.loads((ROOT/'TASKS.json').read_text(encoding='utf8'))['tasks']
    expected={'click_bell','drawer_open_place','handle_basket','item_assembly','items_handover','manipulate_pipette','mixer_operating','sample_loading','table_rearrangement','water_pouring'}
    errors=[]
    if set(mapping)!=expected:errors.append('Task coverage differs from the ten required tasks')
    for task,e in mapping.items():
        paths=[e['config'], e['adapter_directory']+'/__init__.py',e['adapter_directory']+'/deploy_policy.py',*e['paths'].values(),*e['required_assets']]
        for relative in paths:
            p=(ROOT/task/relative).resolve()
            if not p.is_relative_to((ROOT/task).resolve()):errors.append(f'Outside package: {p}')
            elif not p.exists():errors.append(f'Missing: {p}')
    for p in ROOT.rglob('*.py'):
        if any(s in p.parts for s in ('_history','__pycache__','history_prepackage_20261010')):continue
        try:ast.parse(p.read_text(encoding='utf-8-sig'))
        except Exception as exc:errors.append(f'Syntax {p}: {exc}')
    checked=0
    if args.integrity:
        manifest=ROOT/MANIFEST_NAME
        if not manifest.exists():errors.append('SHA256SUMS_UNIFIED missing')
        else:
            for line in manifest.read_text(encoding='utf8').splitlines():
                expected_hash,relative=line.split('  ',1)
                p=(ROOT/relative).resolve()
                if not p.is_relative_to(ROOT):errors.append(f'Outside package: {relative}');continue
                if not p.is_file() or digest(p)!=expected_hash:errors.append(f'Hash mismatch or missing: {relative}')
                checked+=1
            listed={line.split('  ',1)[1] for line in manifest.read_text(encoding='utf8').splitlines() if '  ' in line}
            expected_files={p.relative_to(ROOT).as_posix() for p in manifest_files()}
            for relative in sorted(expected_files-listed):errors.append(f'Unlisted package file: {relative}')
            for relative in sorted(listed-expected_files):errors.append(f'Stale manifest entry: {relative}')
    print(json.dumps({'tasks_checked':len(mapping),'hashes_written':written,'hashes_checked':checked,'errors':errors,'gpu_validation':'not_run'},ensure_ascii=False,indent=2))
    return 1 if errors else 0

if __name__=='__main__':
    sys.exit(main())
