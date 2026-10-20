"""Configuration and delegation checks against a prepared retained tree."""
import importlib
import json
from pathlib import Path
import sys
import types
import yaml
POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY.parent))
from policy import _submission as bridge
entries = json.loads((POLICY/'_nyx_bundle/TASKS.json').read_text())['tasks']
for task, entry in entries.items():
    folder = POLICY / ('nyx_'+task)
    adapter = importlib.import_module('policy.nyx_'+task)
    config = yaml.safe_load((folder/'deploy_policy.yml').read_text())
    resolved = bridge.resolve_config(task, config)
    frozen = yaml.safe_load((bridge.ROOT/task/entry['config']).read_text())
    frozen.update(entry.get('overrides', {}))
    for key, value in frozen.items():
        if key not in entry['paths'] and key not in ('task_name','policy_name','model_name','setting','headless'):
            assert resolved[key] == value, (task,key)
    for key, rel in entry['paths'].items():
        assert Path(resolved[key]) == (bridge.ROOT/task/rel).resolve(), (task,key)
    config.update(seed=451, max_episodes=7)
    config['checkpoint_path'] = str(POLICY/'external explicit checkpoint')
    fake = types.ModuleType('policy._frozen_'+task)
    fake.get_model = lambda args: args
    fake.eval = lambda env, model, obs: (obs, {'ok':True}, False, [0.01])
    fake.reset_model = lambda model: model.update(reset=True)
    sys.modules[fake.__name__] = fake
    model = adapter.get_model(config)
    assert model['seed'] == 451 and model['max_episodes'] == 7
    assert Path(model['checkpoint_path']) == Path(config['checkpoint_path'])
    assert adapter.eval(None, model, 'obs') == ('obs', {'ok':True}, False, [0.01])
    adapter.reset_model(model)
    assert model['reset'] is True
    del sys.modules[fake.__name__]
print('PASS: ten adapter configs, preserved frozen control settings, explicit checkpoint overrides, four-value/timing forwarding and reset forwarding. Model construction uses test doubles; no inference.')
