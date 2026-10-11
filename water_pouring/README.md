# water_pouring submission

Fixed RoboSynChallenge submission bundle for the current best `water_pouring`
policy: spatial-translation fine-tune checkpoint `000200`.

## Evidence and scope

- Development screen: 18/20.
- Independent 100-episode seed-0 run: 81/100.
- All 120 recorded initial RGB hashes were distinct across the two runs.
- Frozen seeds 101, 202 and 303 were not used.
- These are local simulation observations, not a statistical or leaderboard
  guarantee. Frozen final validation remains outstanding.

## Policy contract

- ACT chunk size: 50; execution queue: 50; temporal ensemble: disabled.
- No action offset, retry, recovery, post-processing, or policy switching.
- Inputs: three public RGB cameras plus qpos/qvel/qf.
- No simulator object pose, contact state, task metric, hidden success signal,
  or evaluation label is used to generate actions.
- The public organizer success method is queried only after an environment step
  to stop an already successful episode.

## Manual evaluation from this delivery directory

```bash
cd /home/user/Robotsynchallenge/RoboSynChallenge

bash ../最终提交/water_pouring/eval.sh \
  water_pouring random \
  ../最终提交/water_pouring/checkpoint 0 \
  --max_episodes 20 \
  --seed 1 \
  --headless True \
  --eval_video_log True
```

Use only non-frozen development seeds for ordinary checks. Do not use seeds
101, 202 or 303 until final independent acceptance.

## Standard installation layout

The official custom-policy interface expects a policy directory containing
`__init__.py`, `deploy_policy.py`, `deploy_policy.yml`, and `eval.sh`. This
bundle can also be copied intact to `RoboSynChallenge/policy/water_pouring`.
The included `manual_eval.py` permits evaluation directly from this delivery
directory without modifying shared policy code.

Official references:

- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/your_own_policy.html
- https://edem-ai.github.io/RoboSynChallenge/html/tutorials/policy/act.html
- https://github.com/EDEM-AI/RoboSynChallenge/tree/main/policy/Your_Policy
