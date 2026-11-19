"""Construct the frozen ACT inputs absent from the real-robot interface."""

import torch


def checkpoint_state_means(policy):
    """Cache raw training means before the policy's existing normalization."""
    means = {}
    device = next(policy.parameters()).device
    for name in ("observation.qvel", "observation.qf"):
        feature = policy.config.input_features.get(name)
        if feature is None:
            continue
        buffer_name = "buffer_" + name.replace(".", "_")
        stats = getattr(policy.normalize_inputs, buffer_name, None)
        if stats is None or "mean" not in stats:
            raise ValueError(f"Frozen checkpoint does not provide a mean for {name}")
        mean = stats["mean"].detach()
        if tuple(mean.shape) != tuple(feature.shape) or not torch.isfinite(mean).all():
            raise ValueError(f"Invalid frozen checkpoint mean for {name}")
        means[name] = mean.to(device=device, dtype=torch.float32).unsqueeze(0)
    return means
