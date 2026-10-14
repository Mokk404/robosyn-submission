# Preparation correction: eval_50aaeafd932748

Date: 2026-10-14

The previous code snapshot contained task implementations in separate task
packages, but lacked a discoverable repository-root policy tree. This correction
adds ten standard adapters under policy/nyx_<task>/, each with __init__.py,
deploy_policy.py, deploy_policy.yml, and eval.sh. The root launcher now uses
these same on-disk adapters. README.md and ENTRYPOINTS.md document all ten
entrypoints, dependency setup, and immutable checkpoint URLs.

Original task implementations, model checkpoints, and action transformations
are preserved. The adapter bridge resolves paths and forwards get_model, eval,
and reset_model to the existing task packages. Tests cover task isolation,
configuration preservation, caller overrides, interface forwarding, and failed
import cleanup. The asset verifier confirms all 396 runtime assets.

Validation is structural and CPU-only; no fresh GPU simulator rollout is claimed.
Separate provenance, qvel/qf, and sample-loading FK/IK questions remain disclosed.
The correction response should reference the immutable GitHub commit containing
this file, plus all ten checkpoint URLs in README.md. Do not use a moving branch
URL as the immutable code reference.
