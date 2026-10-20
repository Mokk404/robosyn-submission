# RoboSynChallenge policy-tree packaging correction

For eval_50aaeafd932748, 2026-10-20. The complete runnable source and setup
helpers are under [policy/README.md](policy/README.md). The preparation process
may retain only policy/. Participant-root files are historical compatibility
records and are not used by the corrected policy-tree entrypoints.

Copy/merge policy/ into the official repository; reuse the existing pinned
checkpoint downloads with policy/prepare_assets.py. No checkpoint contents,
model logic, preprocessing, seeds, action bounds or timing behavior are changed.
