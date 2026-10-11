# Official submission checklist

Updated 2026-10-11. Items marked **TEAM** require facts, credentials, publishing, or organizer decisions that are not available inside this directory.

## Deadline and access

- [ ] **TEAM — urgent:** confirm the exact October 11 preliminary-round cutoff time with the organizer.
- [ ] **TEAM:** confirm that the participant email and access token work on the competition portal.
- [ ] **TEAM:** record full name, affiliation, email, team name and intended use in the registration record.
- [ ] **TEAM:** send or adapt `ORGANIZER_CONTACT_DRAFT.md` for the unresolved portal/runtime questions.

## Publish the artifacts

- [ ] **TEAM:** create the GitHub code repository and fill `code_url`.
- [ ] **TEAM:** create the Hugging Face model repository and fill `huggingface_checkpoint_url`.
- [ ] Upload every file listed by `ASSETS.json` to Hugging Face at the same relative path.
- [ ] Keep model weights out of ordinary GitHub history. The ten primary safetensors files are about 197 MB each.
- [ ] Test a clean reconstruction with:

  ```bash
  python -m pip install huggingface_hub
  python download_assets.py --repo-id ORG/REPO --revision PINNED_REVISION
  python -B verify_submission.py --integrity
  ```

- [ ] Pin and record a GitHub commit and Hugging Face revision rather than relying only on mutable `main` branches.
- [ ] **TEAM:** select a license for original submission code and add third-party notices where required. No license is invented by this package.

## Resolve portal protocol blockers

- [ ] **TEAM / ORGANIZER:** decide whether the ten task-specific controllers are one base-model entry or multiple model/evaluation entries.
- [ ] **TEAM / ORGANIZER:** resolve the portal rule `execution_chunk_size < output_chunk_size` for the six ACT policies that truthfully use `50/50`.
- [ ] **TEAM / ORGANIZER:** confirm how the non-ACT `sample_loading` controller should declare output/execution chunk sizes.
- [ ] **TEAM:** confirm the gripper threshold requested by the portal. Do not assume the portal default represents the physical joint targets emitted by these adapters.
- [ ] **TEAM / ORGANIZER:** obtain written approval for `sample_loading` access to `env.unwrapped.robot` FK/IK, or replace it with an approved public interface and revalidate.
- [ ] **TEAM / ORGANIZER:** confirm whether qvel/qf are present in the official observations for policies that consume them.

Prepared portal values and unresolved fields are recorded in `PORTAL_FIELDS.json`.

## Complete data disclosure

- [ ] **TEAM:** select official real, official simulated and/or other data truthfully.
- [ ] **TEAM:** if `other` is selected, describe every custom simulated trajectory set, RGB regressor dataset, RBF prior and fixed schedule source.
- [ ] Confirm dataset revisions, actual usable episode counts and redistribution rights where `DATA_SOURCES.md` currently says they are unverified.
- [ ] Do not describe runtime trajectory banks as model weights; disclose them as runtime assets and explain how they were derived.

## Official runtime acceptance

- [ ] Pin the official RoboSynChallenge commit and record EmbodiChain, DexSim, Python, PyTorch, CUDA, driver and GPU versions.
- [ ] On the official Linux host, run one real model-loaded episode for every task.
- [ ] Confirm successful model load, four-result policy return, 14-dimensional action, clean exit, video output and `evaluation_metrics.json`.
- [ ] Run the chosen multi-episode/random protocol after all one-episode smoke tests pass.
- [ ] Fill `OFFICIAL_RUNTIME_VALIDATION.json` with evidence paths and results.
- [ ] Re-run integrity checks after the final code/HF reconstruction.

## Portal submission

- [ ] Register the base model name, Hugging Face link and GitHub link.
- [ ] Complete every field listed in `PORTAL_FIELDS.json` after blockers are resolved.
- [ ] **TEAM:** choose ranked or test run.
- [ ] Submit the evaluation request and save the resulting submission ID/status screenshot or JSON record.

