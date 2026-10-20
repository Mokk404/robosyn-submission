# manipulate_pipette: official rollout evidence still required

The E-drive package fixes the four-result policy return contract and disables
the local-only interval-event freeze. Static checks and a CPU timing mock
passed. **No fresh official-runtime rollout of this exact package and config
has been completed.** The historical 57/100 Pure ACT result used the prior
local evaluation configuration and is not a fresh result for this package.

After making the E-drive package visible to the Linux simulator:

```bash
export ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge
export PYTHON_BIN="$ROBOSYNCHALLENGE_ROOT/policy/act/.venv/bin/python"
cd /path/to/manipulate_pipette
bash deploy/eval.sh 0 1 0
```

Verify the log selects `official4`, loads current1500 with the recorded
SHA256, performs one valid action, returns inference timing and exits normally.
Then run the same entry with `0 100 0`, retain all 100 per-episode results
and the final summary. Until that completes, official success for this exact
entry is **unverified**. The simulator and evaluator must remain unchanged;
no teacher, press guard, queue refresh or privileged object state is allowed.
