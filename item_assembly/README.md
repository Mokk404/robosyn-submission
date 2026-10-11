# item_assembly_1

Final fixed submission package for the first `item_assembly` route. It combines
the original ACT checkpoint with a fixed, bounded qpos-conditioned correction.

## Install and evaluate

Copy this complete directory to `RoboSynChallenge/policy/item_assembly_1`, then:

```bash
cd /home/user/Robotsynchallenge/RoboSynChallenge
bash policy/item_assembly_1/eval.sh item_assembly random policy/item_assembly_1/checkpoint 0
```

Extra evaluator overrides may be appended, for example `--max_episodes 100`.
The official evaluator dynamically imports `policy.item_assembly_1`, so the
directory name and location must not be changed after installation.

## Fixed policy contract

- ACT checkpoint: original `080000` model, included in `checkpoint/`.
- Model SHA-256: `ff69fa981b78bb55a8ad7e82fccdd24b67bd9938aabf1b37959ca4e192edbd0c`.
- Residual SHA-256: `d79b8abfb5e7556d804165d11a0e1950fe2d687ead96107c4a127f4e30e0b611`.
- ACT output chunk: 50; executed before replanning: 20 (`20 < 50`).
- Action offset: 0; temporal ensembling: disabled.
- Residual input: public robot qpos only (`observation.state`).
- Residual output: twelve arm joints only; both gripper dimensions are unchanged.
- Per-joint residual bound: 0.02 rad; parameters stay fixed for the full run.
- Runtime observations: three public RGB cameras plus qpos/qvel/qf. No object
  pose, contact state, IK truth, simulator state, or privileged success signal
  is used to generate actions.

The policy only calls the organizer's public `is_task_success` method after an
environment step to stop an already successful episode, matching the official
deployment adapter.

## Validation record

The frozen first-route policy scored 81/100 and 84/100 in two independent
100-episode acceptance runs. These are local simulation results, not a promise
of leaderboard performance.

## Official interface references

- https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/Your_Policy
- https://github.com/EDEM-AI/RoboSynChallenge/blob/main/scripts/eval_policy.py
- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html

