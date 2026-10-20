import unittest
from types import SimpleNamespace
from collections import deque
import numpy as np
import torch
from policy.act_table_physical_v1.deploy_policy import encode_action, encode_obs, eval, reset_model


def environment():
    low, high = np.full(14, -10.), np.full(14, 10.)
    low[[6, 13]], high[[6, 13]] = 0., .05
    inner = SimpleNamespace(single_action_space=SimpleNamespace(low=low, high=high, shape=(14,)), device='cpu')
    return SimpleNamespace(unwrapped=inner)


class Tests(unittest.TestCase):
    def test_physical_value_preserved_and_input_unmodified(self):
        action = torch.arange(14).float().unsqueeze(0) / 100
        action[:, [6, 13]] = .04
        saved = action.clone()
        result = encode_action(action, environment())
        torch.testing.assert_close(result, saved)
        torch.testing.assert_close(action, saved)

    def test_batched_clipping_uses_environment_limits(self):
        env = environment()
        env.unwrapped.single_action_space.high[13] = .03
        action = torch.zeros(2, 14)
        action[:, [6, 13]] = torch.tensor([[-.01, .08], [.08, -.01]])
        result = encode_action(action, env)
        torch.testing.assert_close(result[:, [6, 13]], torch.tensor([[0., .03], [.05, 0.]]))

    def test_invalid_action_rejected(self):
        for action in (torch.zeros(1, 13), torch.full((1, 14), float('nan'))):
            with self.assertRaises(ValueError):
                encode_action(action, environment())

    def test_observation_rgb_and_proprioception(self):
        model = SimpleNamespace(config=SimpleNamespace(input_features=['observation.state', 'observation.images.cam_high']), table_device=torch.device('cpu'), table_states={'observation.state':'robot/qpos'}, table_images={'observation.images.cam_high':'cam_high'})
        obs = {'robot':{'qpos':np.ones(14)}, 'sensor':{'cam_high':{'color':np.full((8, 9, 4), 255, dtype=np.uint8)}}}
        batch = encode_obs(obs, model)
        self.assertEqual(tuple(batch['observation.images.cam_high'].shape), (1, 3, 8, 9))
        self.assertEqual(batch['observation.images.cam_high'].max().item(), 1)
        self.assertEqual(tuple(batch['observation.state'].shape), (1, 14))

    def test_loop_sends_correct_action_and_stops_at_truncation(self):
        env = environment()
        sent = []
        def step(action):
            sent.append(action)
            return {}, 0, False, True, {}
        env.step = step
        model = SimpleNamespace(config=SimpleNamespace(input_features=[], temporal_ensemble_coeff=None), _action_queue=deque(), table_device=torch.device('cpu'), table_act_step=10, table_timing=False, select_action=lambda batch:torch.full((1, 14), .04))
        result = eval(env, model, {})
        self.assertEqual(len(result), 3)
        self.assertEqual(result[1]['_policy_action_steps'], 1)
        self.assertEqual(len(sent), 1)
        self.assertAlmostEqual(sent[0][0, 6].item(), .04)
        model.table_timing = True
        self.assertEqual(len(eval(env, model, {})), 4)

    def test_reset_clears_queue(self):
        queue = deque([1])
        reset_model(SimpleNamespace(reset=queue.clear))
        self.assertFalse(queue)


if __name__ == '__main__':
    unittest.main()
