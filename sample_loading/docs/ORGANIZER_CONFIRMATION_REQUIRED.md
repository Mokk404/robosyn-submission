# Organizer confirmation required: robot FK/IK access

Status: **not confirmed**. This document does not assert either compliance or
non-compliance.

The packaged V89 policy uses public RGB and robot observations for perception,
but runtime code obtains the simulator's robot kinematic model through
`env.unwrapped.robot`. Its active dependency chain includes
`compute_fk(...)` in
`_runtime/policy/sample_loading_90plus_visual_v87_public_obs_direct_controller_v1/deploy_policy.py`
and `compute_ik(...)` in
`_runtime/policy/sample_loading_90plus_visual_v85_fresh_rgb_multirecovery_v1/deploy_policy.py`.
This access is distinct from reading object ground-truth poses, but whether
the organizer permits it is not established by the available written rules.

Ask the organizer in writing:

> For sample_loading, may a submitted policy call the robot-model FK/IK
> methods exposed at `env.unwrapped.robot.compute_fk/compute_ik`, using only
> observed robot joint state and RGB-derived targets? No object pose, contact,
> task-success ground truth, or privileged affordance data are read for policy
> control. If this access is not permitted, which documented robot-kinematics
> API, if any, is allowed?

Do not label the current V89 package ranked-upload-ready until the answer is
received and recorded here. The existing random20 evidence (15/20) establishes
observed local performance, not organizer approval of this interface.

Confirmation record (intentionally blank):

- Date and organizer representative:
- Exact written reply / link:
- Permitted API and conditions:
- Required code change or no change:
- Follow-up official-runtime smoke result:
