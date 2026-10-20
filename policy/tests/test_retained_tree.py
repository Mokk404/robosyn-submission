"""Reproduce preparation retaining only policy/, with optional real fixed assets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
POLICY = Path(__file__).resolve().parents[1]

def run(argv, cwd, env):
    result = subprocess.run(argv, cwd=cwd, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(str(argv)+'\n'+result.stdout+'\n'+result.stderr)
    return result.stdout

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', help='Existing unchanged HF snapshot root for full retained-tree test')
    parser.add_argument('--bash', default=shutil.which('bash'))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='nyx20_') as temp:
        official = Path(temp)/'official'; official.mkdir()
        target = official/'policy'
        shutil.copytree(POLICY,target,ignore=shutil.ignore_patterns('__pycache__','.cache'))
        scripts = official/'scripts';scripts.mkdir()
        # Minimal evaluator fixture imports the real submitted adapter through its official name.
        (scripts/'eval_policy.py').write_text(
            'import importlib,sys\n'
            'name=sys.argv[sys.argv.index("--policy_name")+1]\n'
            'adapter=importlib.import_module("policy."+name)\n'
            'assert all(callable(getattr(adapter,k,None)) for k in ("get_model","eval","reset_model"))\n'
            'print("OFFICIAL_IMPORT_FIXTURE_PASS",name)\n',encoding='utf-8')
        tasks=json.loads((target/'_nyx_bundle/TASKS.json').read_text())['tasks']
        assert all(not (official/name).exists() for name in ['TASKS.json','run_task.py','download_assets.py',*tasks])
        env=os.environ.copy()
        for name in ('PYTHONPATH','ROBOSYNCHALLENGE_ROOT','NYX_ASSET_ROOT','NYX_ASSET_MAP'):
            env.pop(name,None)
        env['PYTHONDONTWRITEBYTECODE']='1'
        env['PYTHON_BIN']=Path(sys.executable).as_posix()
        verify=run([sys.executable,'-B',str(target/'verify_bundle.py')],official,env)
        assert not json.loads(verify)['errors']
        if not args.asset_root:
            print('PASS: policy-only source/provenance checks. Supply --asset-root for all asset/launcher tests.')
            return
        download=Path(args.asset_root).resolve()
        locations={task:str(download/task) for task in tasks}
        first='table_rearrangement/'+tasks['table_rearrangement']['required_assets'][-1]
        locations[first]=str(download/first)
        mapfile=Path(temp)/'existing-downloads.json';mapfile.write_text(json.dumps(locations))
        prepared=run([sys.executable,'-B',str(target/'prepare_assets.py'),'--asset-map',str(mapfile),'--copy'],official,env)
        assert json.loads(prepared)['assets_checked']==396
        run([sys.executable,'-B',str(target/'verify_bundle.py'),'--assets'],official,env)
        run([sys.executable,'-B',str(target/'download_assets.py'),'--verify-only'],official,env)
        run([sys.executable,'-B',str(target/'tests/test_adapters.py')],Path(temp),env)
        for task,entry in tasks.items():
            cmd=[sys.executable,'-B',str(target/'run_task.py'),'--task',task,'--episodes','1','--seed','123']
            plan=json.loads(run(cmd+['--dry-run','--','--eval_video_log','False'],Path(temp),env))
            assert plan['argv'][-2:]==['--eval_video_log','False']
            assert Path(plan['adapter']).is_relative_to(target)
            for key in entry['paths']:
                value=plan['argv'][plan['argv'].index('--'+key)+1]
                assert Path(value).is_relative_to(target),(task,key,value)
            executed=run(cmd,Path(temp),env)
            assert 'OFFICIAL_IMPORT_FIXTURE_PASS nyx_'+task in executed
            if args.bash:
                launcher=str(target/('nyx_'+task)/'eval.sh')
                run([args.bash,'-n',launcher],Path(temp),env)
                plan=json.loads(run([args.bash,launcher,'--episodes','1','--seed','123','--dry-run'],Path(temp),env))
                assert plan['task']==task
        # Reject changed downloaded bytes instead of silently using different weights.
        sys.path.insert(0,str(target))
        from _nyx_bundle import assets
        bad=Path(temp)/'bad.json';bad.write_text('{}')
        badmap=Path(temp)/'bad-map.json';badmap.write_text(json.dumps({first:str(bad)}))
        try: assets.stage('table_rearrangement',asset_map=badmap,verify_only=True)
        except FileNotFoundError: pass
        else: raise AssertionError('Changed pinned file accepted')
        print('PASS: retained policy/ only; no participant root files; 396 identical existing assets mapped; all 10 launch plans, official-import fixtures and bash entrypoints; explicit checkpoints and frozen config parity; wrong-asset rejection. No model inference or simulator rollout.')
if __name__ == '__main__':
    main()
