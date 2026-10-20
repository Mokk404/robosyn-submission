# sample_loading — submission card

## 1. Method and selected artifact

V89 RGB perception and public-observation controller; FK/IK confirmation pending. Original weights and inference/action logic are preserved.

- Adapter: `./__init__.py` exports get_model, eval and reset_model.
- Configuration: `deploy_policy.yml`.
- Primary checkpoint: `checkpoint/pose_regressor_v4.ckpt`.
- Full required assets and SHA-256: ../ASSETS.json (task `sample_loading`).
- Hosted checkpoint URL: **TO BE FILLED**. Upload all required assets with the same relative paths.

## 2. Inputs and outputs

Three RGB cameras and robot qpos/gripper. Drawer neutralizes legacy qvel/qf; sample_loading additionally uses robot FK/IK, awaiting written organizer confirmation.
Output: 14 absolute joint/gripper targets. Preserve the adapter's existing units and scaling. Table uses physical gripper clipping; no global gripper transformation is applied by the unified runner.
The package entry exports the official four-result interface. click_bell and manipulate_pipette use the existing official_contract wrapper around their legacy executor.

## 3. Training data and provenance

Existing README declares simulated demonstrations, selected/self-generated simulated trajectories and visual pose regressors. Runtime trajectory banks are included under _runtime. Training-only pose labels are distinguished from runtime inputs; exact training split/revisions remain to be confirmed.

## 4. Dependencies and command

See ../ENVIRONMENT.md and requirements-runtime.txt. From the **submission root**, after configuring the official Linux runtime:

```bash
bash eval.sh --task sample_loading --gpu 0 --episodes 1 --seed 0
bash eval.sh --task sample_loading --gpu 0 --episodes 100 --seed 0
```

Add `--dry-run` to inspect paths and arguments without loading models or running the simulator. Official evaluator overrides go after `--`, for example `-- --eval_video_log True`.

## 5. Evaluation evidence and limitations

Existing task README, provenance and evidence describe historical runs. They are not new results for this unified launcher and are not organizer-held-out scores.
Written organizer confirmation of env.unwrapped.robot FK/IK access remains outstanding.

## 6. Submission status

Local assets are inventoried. GitHub/Hugging Face links, owner/team identity and the leaderboard choice remain external fields in ../SUBMISSION_FORM.json. See ../PENDING.md before submitting.
