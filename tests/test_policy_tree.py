"""Adapter integration checks; no model inference or simulator required."""
import ast
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from policy import _submission as bridge
records = json.loads((ROOT / 'TASKS.json').read_text(encoding='utf-8'))['tasks']
with tempfile.TemporaryDirectory(prefix='policy cwd ') as tmp:
    previous = Path.cwd()
    os.chdir(tmp)
    try:
        for task, record in records.items():
            folder = ROOT / 'policy' / ('nyx_' + task)
            for filename in ('__init__.py', 'deploy_policy.py', 'deploy_policy.yml', 'eval.sh'):
                assert (folder / filename).is_file(), (task, filename)
            cfg = yaml.safe_load((folder / 'deploy_policy.yml').read_text())
            adapter = importlib.import_module('policy.' + cfg['policy_name'])
            resolved = bridge.resolve_config(task, cfg)
            for key, relative in record['paths'].items():
                assert Path(resolved[key]) == (ROOT / task / relative).resolve()
                assert Path(resolved[key]).exists()
            original = yaml.safe_load((ROOT / task / record['config']).read_text())
            original.update(record.get('overrides', {}))
            for key, value in original.items():
                if key not in record['paths'] and key not in ('task_name','policy_name','model_name','setting','headless'):
                    assert resolved[key] == value, (task,key)
            cfg['seed'] = 451
            cfg['max_episodes'] = 7
            cfg['checkpoint_path'] = str(Path(tmp) / 'external checkpoint')
            fake = types.ModuleType('policy._frozen_' + task)
            fake.get_model = lambda args: args
            fake.eval = lambda env, model, obs: (obs, {'ok':True}, False, [0.01])
            fake.reset_model = lambda model: model.update(reset=True)
            sys.modules[fake.__name__] = fake
            model = adapter.get_model(cfg)
            assert model['seed'] == 451 and model['max_episodes'] == 7
            assert Path(model['checkpoint_path']) == Path(cfg['checkpoint_path'])
            assert adapter.eval(None, model, 'obs') == ('obs', {'ok':True}, False, [0.01])
            adapter.reset_model(model)
            assert model['reset'] is True
            del sys.modules[fake.__name__]
        try:
            bridge.resolve_config('click_bell', {'task_name':'table_rearrangement'})
        except ValueError:
            pass
        else:
            raise AssertionError('Cross-task configuration accepted')
    finally:
        os.chdir(previous)
    # Real import loader: relative imports, distinct namespaces and retry after failure.
    saved_root = bridge.ROOT
    fixture = Path(tmp) / 'bundle'
    fixture.mkdir()
    fixture_records = {}
    for task in ('one','two'):
        folder = fixture / task / 'submitted_act'; folder.mkdir(parents=True)
        (folder / '__init__.py').write_text('from .deploy_policy import get_model, eval, reset_model\n')
        (folder / 'deploy_policy.py').write_text(
            f'def get_model(args): return {task!r}\ndef eval(*args): return 1,2,False,[]\ndef reset_model(model): pass\n')
        fixture_records[task] = {'adapter_directory':'submitted_act'}
    (fixture / 'TASKS.json').write_text(json.dumps({'tasks':fixture_records}))
    bridge.ROOT = fixture
    try:
        assert bridge.implementation('one').get_model({}) == 'one'
        assert bridge.implementation('two').get_model({}) == 'two'
        sys.modules.pop('policy._frozen_one')
        (fixture / 'one/submitted_act/__init__.py').write_text('raise RuntimeError("fixture failure")\n')
        try: bridge.implementation('one')
        except RuntimeError: pass
        else: raise AssertionError('Import failure suppressed')
        assert 'policy._frozen_one' not in sys.modules
    finally:
        bridge.ROOT = saved_root
        for name in list(sys.modules):
            if name.startswith(('policy._frozen_one','policy._frozen_two')):
                sys.modules.pop(name)
print('PASS: 10 standard policy imports, relative paths from unrelated cwd, frozen control settings, caller overrides, four-value forwarding, reset forwarding, isolated implementations and import failure cleanup. No GPU inference.')
