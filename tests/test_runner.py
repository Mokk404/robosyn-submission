import importlib.util,json,sys,tempfile,types
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('nyx_runner',ROOT/'run_task.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
entries=json.loads((ROOT/'TASKS.json').read_text(encoding='utf8'))['tasks']
portal=json.loads((ROOT/'PORTAL_FIELDS.json').read_text(encoding='utf8'))
assert set(portal['task_protocols'])==set(entries)
assert sum(item['status']=='portal-conflict' for item in portal['task_protocols'].values())==6
assert portal['evaluation']['output']['joints']=={'enabled':True,'delta':False}
assert portal['evaluation']['output']['gripper']=={'enabled':True,'delta':False}
with tempfile.TemporaryDirectory(prefix='nyx test ') as tmp:
    official=Path(tmp)/'official runtime';(official/'scripts').mkdir(parents=True)
    (official/'scripts/eval_policy.py').write_text('# dry-run fixture; no simulator\n')
    for task in entries:
        plan=runner.resolve_plan(task,official,1,123,'0','random',['--eval_video_log','False'])
        assert plan['argv'][plan['argv'].index('--task_name')+1]==task
        assert plan['argv'][plan['argv'].index('--max_episodes')+1]=='1'
        assert plan['argv'][plan['argv'].index('--checkpoint_path')+1]==str((ROOT/task/entries[task]['paths']['checkpoint_path']).resolve())
        assert plan['argv'][-2:]==['--eval_video_log','False']
        assert Path(plan['adapter'],'__init__.py').is_file()
    # Distinct aliases must preserve separate packages even when both original names are submitted_act.
    saved=sys.modules.get('policy');policy=types.ModuleType('policy');policy.__path__=[];sys.modules['policy']=policy
    for index in (1,2):
        package=Path(tmp)/str(index)/'submitted_act';package.mkdir(parents=True)
        (package/'__init__.py').write_text(f'def get_model(args): return {index}\ndef eval(*args): return None,None,False,[]\ndef reset_model(model): pass\n')
        module=runner.register_adapter({'adapter':str(package),'policy_name':f'nyx_test_{index}'})
        assert module.get_model({})==index
    assert sys.modules['policy.nyx_test_1'].get_model({})==1
    for index in (1,2):sys.modules.pop(f'policy.nyx_test_{index}',None)
    if saved is None:sys.modules.pop('policy',None)
    else:sys.modules['policy']=saved

# Exercise the existing wrapper without importing ACT or changing its implementation.
for task in ('click_bell','manipulate_pipette'):
    p=ROOT/task/'policy/submitted_act/official_contract.py'
    spec=importlib.util.spec_from_file_location('contract_'+task,p)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Model:
        act_device='cpu'
        def predict_action_chunk(self,x):return x+1
    model=Model()
    def execute(env,model,obs):
        assert model.predict_action_chunk(2)==3
        return obs,{'ok':True},False
    result=module.eval_with_inference_times(execute,None,model,{'robot':{}})
    assert len(result)==4 and len(result[3])==1 and result[3][0]>=0
    assert 'predict_action_chunk' not in vars(model)
    def fail(*args):raise RuntimeError('expected test exception')
    try:module.eval_with_inference_times(fail,None,model,None)
    except RuntimeError:pass
    else:raise AssertionError('exception suppressed')
    assert 'predict_action_chunk' not in vars(model)

print('PASS: 10 task plans, portal protocol coverage, absolute asset paths, argument forwarding, independent adapter namespaces, 2 four-result wrappers and exception cleanup. No model/GPU execution.')
