"""Official adapter for drawer_open_place; delegates to the frozen task package."""
from policy._submission import implementation, resolve_config
from policy._nyx_bundle.assets import stage
TASK = 'drawer_open_place'

def get_model(usr_args):
    config = resolve_config(TASK, usr_args)
    stage(TASK, config.get('asset_root'), config.get('asset_map'))
    return implementation(TASK).get_model(config)

def eval(env, model, obs):
    return implementation(TASK).eval(env, model, obs)

def reset_model(model):
    return implementation(TASK).reset_model(model)
