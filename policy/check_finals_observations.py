import argparse
import contextlib
import importlib
import io
import os
from pathlib import Path
import sys
import tempfile
from collections import deque
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
TASKS = (
    "click_bell", "handle_basket", "item_assembly", "manipulate_pipette",
    "mixer_operating", "table_rearrangement", "water_pouring",
)
CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")
STATE = "observation.state"
MISSING = ("observation.qvel", "observation.qf")


class MockACT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("device_marker", torch.nn.Parameter(torch.zeros(())))
        vector = SimpleNamespace(shape=(14,))
        images = {f"observation.images.{name}": SimpleNamespace(shape=(3, 8, 8))
                  for name in CAMERAS}
        self.config = SimpleNamespace(
            input_features={STATE: vector, **dict.fromkeys(MISSING, vector), **images},
            image_features=images, chunk_size=50, n_action_steps=50,
            temporal_ensemble_coeff=None,
        )
        self.means = {name: torch.arange(14, dtype=torch.float32) + offset
                      for offset, name in enumerate(MISSING, 1)}
        self.normalize_inputs = SimpleNamespace(**{
            "buffer_" + name.replace(".", "_"): {"mean": mean}
            for name, mean in self.means.items()
        })
        self.predictions = 0
        self.reset()

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def reset(self):
        self._action_queue = deque(maxlen=self.config.n_action_steps)

    def predict_action_chunk(self, batch):
        self.predictions += 1
        return torch.zeros(1, self.config.chunk_size, 14)

    def select_action(self, batch):
        assert set(batch) == set(self.config.input_features)
        for key, mean in self.means.items():
            torch.testing.assert_close(batch[key], mean.unsqueeze(0))
        assert batch[STATE].shape == (1, 14)
        for key in self.config.image_features:
            assert batch[key].shape == (1, 3, 8, 8)
        if not self._action_queue:
            chunk = self.predict_action_chunk(batch)
            self._action_queue.extend(chunk.transpose(0, 1)[:self.config.n_action_steps])
        return self._action_queue.popleft()


class NoMotionEnvironment:
    def __init__(self, device):
        self.device = device
        self.unwrapped = self
        self.single_action_space = SimpleNamespace(
            shape=(14,), low=np.full(14, -1.0), high=np.full(14, 1.0)
        )
        self.steps = 0
        self.image_sizes = {camera: (8, 8) for camera in CAMERAS}

    def observation(self):
        # These fixtures omit both unavailable fields at every step.
        return {
            "robot": {"qpos": torch.full((1, 14), 0.001 * self.steps)},
            "sensor": {name: {"color": np.full((1, *self.image_sizes[name], 3), 64, dtype=np.uint8)}
                       for name in CAMERAS},
        }

    def get_wrapper_attr(self, name):
        assert name == "is_task_success"
        return lambda: False

    def step(self, action):
        assert action.shape == (1, 14) and torch.isfinite(action).all()
        self.steps += 1
        return self.observation(), 0.0, False, False, {}


def install_mock_modules():
    for name in ("lerobot", "lerobot.configs", "lerobot.configs.policies",
                 "lerobot.policies", "lerobot.policies.act",
                 "lerobot.policies.act.modeling_act"):
        sys.modules[name] = ModuleType(name)
    sys.modules["lerobot.configs.policies"].PreTrainedConfig = object
    sys.modules["lerobot.policies.act.modeling_act"].ACTPolicy = MockACT


def run(task, checkpoint, args):
    from policy._submission import implementation, resolve_config

    adapter = importlib.import_module(f"policy.nyx_{task}.deploy_policy")
    impl = implementation(task)
    if args.mock:
        adapter.stage = lambda *_args, **_kwargs: None
        impl_module = sys.modules[impl.get_model.__module__]
        if task in ("item_assembly", "water_pouring"):
            # Mock only: there is no real asset to hash. Real mode executes
            # the original integrity checks and assembly residual installation.
            impl_module._sha256 = lambda _path: impl_module.MODEL_SHA256
        if task == "item_assembly":
            def skip_residual(policy, _path):
                policy.item_assembly_residual_sha256 = "mock"
            impl_module._install_residual = skip_residual

    options = {"device": args.device, "pytorch_device": args.device,
               "debug_button_press": False, "debug_trajectory": False}
    if args.mock:
        options["checkpoint_path"] = str(checkpoint)
    config = resolve_config(task, options)
    env = NoMotionEnvironment(torch.device(args.device))
    with torch.inference_mode(), contextlib.redirect_stdout(io.StringIO()):
        model = adapter.get_model(config)
        camera_map = getattr(model, "table_images", None) or model.image_key_map
        for key, feature in model.config.image_features.items():
            env.image_sizes[camera_map[key]] = tuple(feature.shape[-2:])
        adapter.reset_model(model)
        obs = env.observation()
        while env.steps <= model.config.n_action_steps:
            previous_steps = env.steps
            obs = adapter.eval(env, model, obs)[0]
            assert env.steps > previous_steps
        if args.mock:
            assert model.predictions >= 2
        before = env.steps
        adapter.reset_model(model)
        adapter.eval(env, model, env.observation())
        assert env.steps > before
        if args.mock:
            assert model.predictions >= 3
    print(f"PASS {task}: {env.steps} action steps including reset")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--asset-map", type=Path)
    parser.add_argument("--task", choices=TASKS)
    args = parser.parse_args()
    if args.asset_root:
        os.environ["NYX_ASSET_ROOT"] = str(args.asset_root.resolve())
    if args.asset_map:
        os.environ["NYX_ASSET_MAP"] = str(args.asset_map.resolve())
    if args.mock:
        install_mock_modules()
    with tempfile.TemporaryDirectory(dir=Path.cwd() if args.mock else None) as tmp:
        checkpoint = Path(tmp)
        if args.mock:
            (checkpoint / "model.safetensors").touch()
        for task in (args.task,) if args.task else TASKS:
            run(task, checkpoint, args)


if __name__ == "__main__":
    main()
