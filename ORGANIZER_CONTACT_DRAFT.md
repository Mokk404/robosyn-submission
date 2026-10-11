# Organizer contact draft

Recipient: robosynchallenge@gmail.com

Subject: Urgent submission protocol clarification — task-specific chunks and sample_loading interface

Dear RoboSynChallenge organizers,

We are preparing our preliminary-round submission for all ten official tasks and would appreciate urgent clarification on the current portal contract:

1. Our release is a single ten-task suite with task-specific controllers and checkpoints. Should it be registered as one base model/evaluation request, or as multiple model entries?
2. Six ACT controllers produce an output chunk of 50 and execute all 50 actions. The portal currently requires `execution_chunk_size < output_chunk_size`, so the truthful 50/50 declaration cannot be submitted. How should these policies be declared? We do not want to change the tested policy behavior merely to pass form validation.
3. Our `sample_loading` controller is non-ACT and uses `env.unwrapped.robot` FK/IK. Is this interface permitted in the official evaluation environment, and what chunk fields should be entered for this controller?
4. Several ACT controllers consume qpos, qvel and qf. Are qvel and qf available in the official evaluation observations?
5. Please confirm the exact timezone and cutoff time for the October 11 preliminary deadline, and the supported access method if our GitHub or Hugging Face repositories are private.

We can provide the code repository, Hugging Face model repository, per-task asset manifest and SHA-256 checksums immediately after receiving your guidance.

Best regards,

[Full name]

[Team name]

[Affiliation]

[Registered email]

