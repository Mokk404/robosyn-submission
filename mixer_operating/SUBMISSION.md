# mixer_operating — submission card

## 1. Method and selected artifact

ACT retained baseline 100000. Original weights and inference/action logic are preserved.

- Adapter: `policy/act_mixer_submission_v1/__init__.py` exports get_model, eval and reset_model.
- Configuration: `policy/act_mixer_submission_v1/deploy_policy.yml`.
- Primary checkpoint: `policy/act_mixer_submission_v1/checkpoint`.
- Full required assets and SHA-256: ../ASSETS.json (task `mixer_operating`).
- Hosted checkpoint URL: **TO BE FILLED**. Upload all required assets with the same relative paths.

## 2. Inputs and outputs

Three RGB cameras (cam_high, cam_left_wrist, cam_right_wrist), robot qpos/qvel/qf.
Output: 14 absolute joint/gripper targets. Preserve the adapter's existing units and scaling. Table uses physical gripper clipping; no global gripper transformation is applied by the unified runner.
The package entry exports the official four-result interface. click_bell and manipulate_pipette use the existing official_contract wrapper around their legacy executor.

## 3. Training data and provenance

Related project discussion records official1000_custom100 simulated data; this is a planned/named mixture, not a verified count. The retained checkpoint is 100000. Dataset revision and actual valid episode count remain unverified.

## 4. Dependencies and command

See ../ENVIRONMENT.md and requirements-runtime.txt. From the **submission root**, after configuring the official Linux runtime:

```bash
bash eval.sh --task mixer_operating --gpu 0 --episodes 1 --seed 0
bash eval.sh --task mixer_operating --gpu 0 --episodes 100 --seed 0
```

Add `--dry-run` to inspect paths and arguments without loading models or running the simulator. Official evaluator overrides go after `--`, for example `-- --eval_video_log True`.

## 5. Evaluation evidence and limitations

Existing task README, provenance and evidence describe historical runs. They are not new results for this unified launcher and are not organizer-held-out scores.
No new real-checkpoint load or GPU simulation was run during this uniform-format update.

## 6. Submission status

Local assets are inventoried. GitHub/Hugging Face links, owner/team identity and the leaderboard choice remain external fields in ../SUBMISSION_FORM.json. See ../PENDING.md before submitting.
