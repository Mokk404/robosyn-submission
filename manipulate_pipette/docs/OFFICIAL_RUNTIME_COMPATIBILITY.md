# Official evaluator compatibility

The original source-project `policy/act/deploy_policy.py` returns three values:
`(obs, info, truncated)`. The official clean evaluator requires four:
`(obs, info, truncated, inference_times)`. The submitted `policy/submitted_act/`
package now exports the official four-result interface by default. Its copied
`deploy_policy.py` remains byte-identical to the source adapter. The additional
`official_contract.py` temporarily times only genuine ACT
`predict_action_chunk()` calls and restores the method after each evaluation
call. Cached queue actions do not create false inference-time entries.

`deploy/eval.sh` checks the selected RoboSynChallenge evaluator source and
sets `SUBMITTED_ACT_EVAL_CONTRACT=official4` for the four-result evaluator or
`local3` for the previously modified local evaluator. Unknown interfaces cause
the launcher to stop. `deploy/run_eval.py` selects the corresponding adapter;
direct import of the submitted policy package uses the official four-result
interface. Neither path changes network weights, normalization, action queue,
action conversion, or task-success logic.

The submission config sets `eval_freeze_interval_events: false`. The local
evaluator's `true` mode rewrites `interval` randomization events to `reset`,
while the inspected official clean evaluator has no such rewriting option.
Disabling it avoids that known local-only test-condition change. Historical
rollout numbers obtained with the earlier setting are not fresh evidence for
this official-aligned configuration.

Static source checks and CPU mock checks can verify return shape, timings,
restoration, and configuration. They are **not** a simulator rollout. A real
submission-entry smoke test in the target official runtime, followed by a
full-task evaluation, remains required before claiming official performance.
