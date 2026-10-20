"""Load frozen task implementations without relocating their relative assets."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent / '_nyx_bundle'

def entry(task):
    return json.loads((ROOT / 'TASKS.json').read_text(encoding='utf-8'))['tasks'][task]

def resolve_config(task, supplied=None):
    import yaml
    record = entry(task)
    config = yaml.safe_load((ROOT / task / record['config']).read_text(encoding='utf-8'))
    config.update(record.get('overrides', {}))
    config.update({'task_name': task, 'policy_name': 'nyx_' + task,
                   'model_name': record['model_name'], 'setting': 'random', 'headless': True})
    config.update(supplied or {})
    if config.get('task_name') != task:
        raise ValueError(f'{task} adapter cannot run task {config.get("task_name")}')
    for key, relative in record['paths'].items():
        value = (supplied or {}).get(key)
        path = Path(value).expanduser() if value else ROOT / task / relative
        if not path.is_absolute():
            path = ROOT / path
        config[key] = str(path.resolve())
    return config

def implementation(task):
    record = entry(task)
    name = 'policy._frozen_' + task
    if name in sys.modules:
        return sys.modules[name]
    for relative in record.get('extra_import_paths', []):
        path = str(ROOT / task / relative)
        if path not in sys.path:
            sys.path.insert(0, path)
    package = ROOT / task / record['adapter_directory']
    spec = importlib.util.spec_from_file_location(
        name, package / '__init__.py', submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {task} from {package}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        for symbol in ('get_model', 'eval', 'reset_model'):
            if not callable(getattr(module, symbol, None)):
                raise TypeError(f'{name} must export {symbol}')
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
