# items_handover — submission card

## 1. Method and selected artifact

Dual ACT, RGB translation predictor, fixed gripper schedule and RBF prior v60. Original weights and inference/action logic are preserved.

- Adapter: `policy/items_handover/__init__.py` exports get_model, eval and reset_model.
- Configuration: `policy/items_handover/deploy_policy.yml`.
- Primary checkpoint: `policy/items_handover/assets/primary_act`.
- Full required assets and SHA-256: ../ASSETS.json (task `items_handover`).
- Hosted checkpoint URL: **TO BE FILLED**. Upload all required assets with the same relative paths.

## 2. Inputs and outputs

Three RGB cameras and robot qpos/gripper; legacy qvel/qf are not runtime observations.
Output: 14 absolute joint/gripper targets. Preserve the adapter's existing units and scaling. Table uses physical gripper clipping; no global gripper transformation is applied by the unified runner.
The package entry exports the official four-result interface. click_bell and manipulate_pipette use the existing official_contract wrapper around their legacy executor.

## 3. Training data and provenance

Method includes dual ACT weights, RGB pose regression, an RBF trajectory prior and a fixed gripper schedule. See packaged gripper_schedule_sources.json and BUILD_PROVENANCE.json. Dataset declarations must cover every component; complete dataset revisions are not established here.

## 4. Dependencies and command

See ../ENVIRONMENT.md and requirements-runtime.txt. From the **submission root**, after configuring the official Linux runtime:

```bash
bash eval.sh --task items_handover --gpu 0 --episodes 1 --seed 0
bash eval.sh --task items_handover --gpu 0 --episodes 100 --seed 0
```

Add `--dry-run` to inspect paths and arguments without loading models or running the simulator. Official evaluator overrides go after `--`, for example `-- --eval_video_log True`.

## 5. Evaluation evidence and limitations

Existing task README, provenance and evidence describe historical runs. They are not new results for this unified launcher and are not organizer-held-out scores.
No new real-checkpoint load or GPU simulation was run during this uniform-format update.

## 6. Submission status

Local assets are inventoried. GitHub/Hugging Face links, owner/team identity and the leaderboard choice remain external fields in ../SUBMISSION_FORM.json. See ../PENDING.md before submitting.
