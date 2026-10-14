"""Official adapter for drawer_open_place; delegates to the frozen task package."""
from policy._submission import implementation, resolve_config
TASK = 'drawer_open_place'

def get_model(usr_args):
    return implementation(TASK).get_model(resolve_config(TASK, usr_args))

def eval(env, model, obs):
    return implementation(TASK).eval(env, model, obs)

def reset_model(model):
    return implementation(TASK).reset_model(model)
