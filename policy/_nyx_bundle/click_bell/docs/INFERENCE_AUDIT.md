# click_bell inference packaging audit

- Selected model source: `outputs/act_click_bell_006_007_merged_bs8_v0/checkpoints/080000/pretrained_model` (three original files copied, no changes).
- Saved arm-specific evaluation logs explicitly name that source checkpoint. The full-task random run is incomplete; its completed-video count is documented separately in README.
- Copied `policy/act/deploy_policy.py` byte-for-byte into `policy/submitted_act/`. Its package initializer now exposes the official four-result `eval` wrapper while retaining the original three-result executor for the modified local evaluator.
- The adapter imports NumPy, PyTorch and LeRobot ACT. It does not import the click_bell expert action bank, task-specific teacher scripts, training dataset or another checkpoint.
- The unmodified RoboSynChallenge `scripts/eval_policy.py` and task environment are runtime dependencies, not bundled copies. The local `deploy/run_eval.py` changes only the adapter import location via the evaluator's loader hook.
- `config/deploy_policy.yml` relocates `policy_name` and sets `eval_freeze_interval_events: false` to avoid the modified local evaluator's interval-to-reset rewrite. The launcher fixes `task_name=click_bell`, `setting=random`, `n_action_steps=10`, `act_step=10`, `max_steps=1000`, and both debug flags false.
- Source adapter's simulator-state `debug_button_press` path is disabled in submitted inference. Gripper policy outputs are mapped once using the environment action-space low/high values.
- Static inspection and source/copied file SHA comparison are packaging checks, not a new rollout or a claim of 100-episode performance.
