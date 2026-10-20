"""Load this bundle as policy.water_pouring, then run the official evaluator."""

import importlib.util
import os
from pathlib import Path
import runpy
import sys


HERE = Path(__file__).resolve().parent


def repo_root():
    configured = os.environ.get("ROBOSYNCHALLENGE_ROOT")
    candidates = [Path(configured)] if configured else []
    candidates.extend([HERE.parents[1] / "RoboSynChallenge", HERE.parents[1]])
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if (candidate / "scripts/eval_policy.py").is_file():
            return candidate
    raise RuntimeError("Cannot locate RoboSynChallenge; set ROBOSYNCHALLENGE_ROOT.")


def main():
    root = repo_root()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root))
    import policy

    name = "policy.water_pouring"
    spec = importlib.util.spec_from_file_location(
        name, HERE / "__init__.py", submodule_search_locations=[str(HERE)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    runpy.run_path(str(root / "scripts/eval_policy.py"), run_name="__main__")


if __name__ == "__main__":
    main()
