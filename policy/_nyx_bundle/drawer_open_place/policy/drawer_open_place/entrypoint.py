"""Expose the packaged policy to an unchanged RoboSynChallenge evaluator."""

import os
import runpy
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
os.chdir(root)
sys.path.insert(0, str(root))
import policy  # noqa: E402

package_policy_root = Path(__file__).resolve().parent.parent
policy.__path__ = [str(package_policy_root), *policy.__path__]
sys.argv = [str(root / "scripts" / "eval_policy.py"), *sys.argv[2:]]
runpy.run_path(sys.argv[0], run_name="__main__")
