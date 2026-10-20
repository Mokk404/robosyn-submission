"""Map unchanged pinned runtime assets into the retained bundle."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent

def records(task=None):
    payload = json.loads((ROOT / 'ASSETS.json').read_text(encoding='utf-8'))
    result = []
    for name, assets in payload.items():
        if task is not None and task != name:
            continue
        for item in assets:
            item = dict(item)
            item['path'] = name + '/' + item['path']
            result.append(item)
    if task is not None and task not in payload:
        raise ValueError(f'Unknown task: {task}')
    return result

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def valid(path, record):
    return (path.is_file() and path.stat().st_size == record['bytes']
            and digest(path) == record['sha256'])

def mapping(path=None):
    path = path or os.environ.get('NYX_ASSET_MAP')
    if not path:
        return {}
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise TypeError('Asset map must be a JSON object')
    result = {}
    for key, value in data.items():
        p = Path(value).expanduser()
        result[key] = p if p.is_absolute() else path.parent / p
    return result

def stage(task=None, asset_root=None, asset_map=None, copy=False, verify_only=False):
    """No network, no policy execution; verify bytes before linking or copying."""
    configured = asset_root or os.environ.get('NYX_ASSET_ROOT')
    external = Path(configured).expanduser().resolve() if configured else None
    locations = mapping(asset_map)
    chosen = records(task)
    errors = []
    linked = copied = 0
    for record in chosen:
        relative = record['path']
        parts = Path(relative).parts
        if Path(relative).is_absolute() or '..' in parts:
            raise ValueError(f'Invalid asset path: {relative}')
        destination = ROOT / relative
        task_name, rest = relative.split('/', 1)
        if relative in locations:
            source = locations[relative]
        elif task_name in locations:
            source = locations[task_name] / rest
        elif external:
            source = external / relative
        else:
            source = destination
        if not valid(source, record):
            errors.append(f'Missing or changed pinned asset: {source}')
            continue
        if verify_only:
            continue
        if destination.exists() or destination.is_symlink():
            if not valid(destination, record):
                errors.append(f'Existing bundle asset differs from pinned manifest: {destination}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if copy:
            shutil.copyfile(source, destination)
            copied += 1
        else:
            try:
                destination.symlink_to(source.resolve())
                linked += 1
            except OSError as exc:
                raise OSError('Cannot create asset link; rerun prepare_assets.py with --copy') from exc
    if errors:
        raise FileNotFoundError('\n'.join(errors) + '\nUse policy/prepare_assets.py or policy/download_assets.py; see policy/README.md.')
    return {'assets_checked': len(chosen), 'linked': linked, 'copied': copied,
            'verification_only': verify_only, 'errors': []}
