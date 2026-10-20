# click_bell: official rollout evidence still required

The E-drive package fixes the four-result policy return contract and disables
the local-only interval-event freeze. Static checks and a CPU timing mock
passed. **No fresh official-runtime rollout of this exact package and config
has been completed.** In particular, the previously interrupted random100
cannot be reported as a completed score.

After making the E-drive package visible to the Linux simulator:

```bash
export ROBOSYNCHALLENGE_ROOT=/path/to/official/RoboSynChallenge
export PYTHON_BIN="$ROBOSYNCHALLENGE_ROOT/policy/act/.venv/bin/python"
cd /path/to/click_bell
bash deploy/eval.sh 0 1 0
```

Verify the log selects `official4`, loads checkpoint `080000` with the
recorded SHA256, performs one valid action, returns inference timing and exits
normally. Then run the same entry with `0 100 0`, retain all 100 per-episode
results and the final summary. Until that completes, official success for
this exact entry is **unverified**. The simulator and evaluator must remain
unchanged; no private object state or local random-event rewrite is permitted.
