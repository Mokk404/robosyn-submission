"""ACT submission adapter: official four-result evaluator interface."""

from .deploy_policy import eval as _legacy_eval
from .deploy_policy import get_model, reset_model
from .official_contract import eval_with_inference_times


def eval(env, model, obs):
    return eval_with_inference_times(_legacy_eval, env, model, obs)
