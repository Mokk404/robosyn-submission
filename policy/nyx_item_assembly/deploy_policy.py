"""Official adapter for item_assembly; delegates to the frozen task package."""
from policy._submission import implementation, resolve_config
TASK = 'item_assembly'

def get_model(usr_args):
    return implementation(TASK).get_model(resolve_config(TASK, usr_args))

def eval(env, model, obs):
    return implementation(TASK).eval(env, model, obs)

def reset_model(model):
    return implementation(TASK).reset_model(model)
