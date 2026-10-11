"""Adapt the unchanged three-result ACT executor to the official evaluator.

LeRobot ACT runs a new network inference inside ``predict_action_chunk`` only
when its action queue is empty. Timing that method excludes cached queue pops.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable


def _synchronize_if_cuda(model: Any) -> None:
    device = getattr(model, "act_device", None)
    if device is not None and str(device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(device)


def eval_with_inference_times(
    legacy_eval: Callable[..., tuple[Any, Any, Any]], env: Any, model: Any, obs: Any
) -> tuple[Any, Any, Any, list[float]]:
    """Return official ``(obs, info, truncated, inference_times)``.

    The original executor and its action conversions remain byte-identical.
    The temporary method wrapper is always removed, including on failure.
    """

    inference_times: list[float] = []
    original_predict = model.predict_action_chunk
    marker = object()
    previous_override = vars(model).get("predict_action_chunk", marker)

    def timed_predict(*args: Any, **kwargs: Any) -> Any:
        _synchronize_if_cuda(model)
        start = perf_counter()
        try:
            return original_predict(*args, **kwargs)
        finally:
            _synchronize_if_cuda(model)
            inference_times.append(float(perf_counter() - start))

    setattr(model, "predict_action_chunk", timed_predict)
    try:
        result = legacy_eval(env, model, obs)
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("ACT executor must return (obs, info, truncated)")
        final_obs, info, truncated = result
        return final_obs, info, truncated, inference_times
    finally:
        if previous_override is marker:
            delattr(model, "predict_action_chunk")
        else:
            setattr(model, "predict_action_chunk", previous_override)
