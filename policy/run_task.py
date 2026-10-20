"""Run a bundled task with an unchanged official RoboSynChallenge evaluator."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys

POLICY_ROOT = Path(__file__).resolve().parent
ROOT = POLICY_ROOT / '_nyx_bundle'
sys.path.insert(0, str(POLICY_ROOT))
from _nyx_bundle.assets import stage


def resolve_plan(task, official_root, episodes=100, seed=0, gpu='0', setting='random', extra=()):
    entries = json.loads((ROOT / 'TASKS.json').read_text(encoding='utf8'))['tasks']
    if task not in entries:
        raise ValueError(f'Unknown task: {task}')
    if episodes < 1:
        raise ValueError('episodes must be positive')
    entry = entries[task]
    stage(task)
    package = POLICY_ROOT / ('nyx_' + task)
    config = package / 'deploy_policy.yml'
    evaluator = Path(official_root).expanduser().resolve() / 'scripts/eval_policy.py'
    for path in [package / '__init__.py', package / 'deploy_policy.py', config, evaluator]:
        if not path.is_file():
            raise FileNotFoundError(path)
    name = 'nyx_' + task
    overrides = {'task_name': task, 'policy_name': name, 'setting': setting,
                 'model_name': entry['model_name'], 'max_episodes': episodes,
                 'seed': seed, 'headless': True}
    overrides.update(entry.get('overrides', {}))
    for key, value in entry['paths'].items():
        path = (ROOT / task / value).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        overrides[key] = str(path)
    for relative in entry['required_assets']:
        if not (ROOT / task / relative).is_file():
            raise FileNotFoundError(ROOT / task / relative)
    argv = [str(evaluator), '--config', str(config.resolve()), '--overrides']
    for key, value in overrides.items():
        argv.extend(['--' + key, str(value)])
    argv.extend(extra)
    return {'task': task, 'adapter': str(package.resolve()), 'policy_name': name,
            'official_root': str(evaluator.parent.parent), 'gpu': str(gpu),
            'argv': argv, 'extra_import_paths': [str(ROOT / task / p) for p in entry.get('extra_import_paths', [])]}


def register_adapter(plan):
    import policy
    submission_policies = str(POLICY_ROOT)
    if submission_policies not in policy.__path__:
        policy.__path__.append(submission_policies)
    package = Path(plan['adapter'])
    name = 'policy.' + plan['policy_name']
    spec = importlib.util.spec_from_file_location(name, package / '__init__.py', submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    for symbol in ('get_model', 'eval', 'reset_model'):
        if not callable(getattr(module, symbol, None)):
            raise TypeError(f'{name} must export {symbol}')
    setattr(policy, plan['policy_name'], module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--official-root', default=os.environ.get('ROBOSYNCHALLENGE_ROOT', str(POLICY_ROOT.parent)))
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--setting', default='random')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--asset-root', help='Existing fixed HF asset snapshot root')
    parser.add_argument('--asset-map', help='Existing download location map JSON')
    args, extra = parser.parse_known_args()
    if not args.official_root:
        parser.error('Set ROBOSYNCHALLENGE_ROOT or pass --official-root')
    if extra[:1] == ['--']:
        extra = extra[1:]
    if args.asset_root:
        os.environ['NYX_ASSET_ROOT'] = str(Path(args.asset_root).expanduser().resolve())
    if args.asset_map:
        os.environ['NYX_ASSET_MAP'] = str(Path(args.asset_map).expanduser().resolve())
    plan = resolve_plan(args.task, args.official_root, args.episodes, args.seed, args.gpu, args.setting, extra)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ.setdefault('EMBODICHAIN_SIM_EXIT_PROCESS', '0')
    sys.dont_write_bytecode = True
    root = Path(plan['official_root'])
    paths = [str(root), str(root / 'policy'), *plan['extra_import_paths']]
    if os.environ.get('EMBODICHAIN_ROOT'):
        paths.append(os.environ['EMBODICHAIN_ROOT'])
    for path in reversed(paths):
        sys.path.insert(0, path)
    os.chdir(root)
    register_adapter(plan)
    sys.argv = plan['argv']
    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    finally:
        module = sys.modules.get('embodichain.lab.sim.sim_manager')
        manager = getattr(module, 'SimulationManager', None)
        flush = getattr(manager, 'flush_cleanup_queue', None)
        if callable(flush):
            flush()


if __name__ == '__main__':
    main()
