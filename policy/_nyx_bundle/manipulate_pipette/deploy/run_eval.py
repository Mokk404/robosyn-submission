"""Load the byte-identical submitted ACT adapter with the project evaluator.

Only import location is relocated. The model, observation/action conversion,
queue implementation, task environment and success check are unchanged.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "policy"))

import submitted_act  # noqa: E402
from submitted_act import deploy_policy as local_three_result_adapter  # noqa: E402
from scripts import eval_policy as base_eval  # noqa: E402


def load_submitted_adapter(policy_name: str):
    if policy_name != "submitted_act":
        raise ValueError(f"Expected submitted_act, received {policy_name!r}")
    contract = os.environ.get("SUBMITTED_ACT_EVAL_CONTRACT", "official4")
    if contract == "official4":
        adapter = submitted_act
    elif contract == "local3":
        adapter = local_three_result_adapter
    else:
        raise ValueError(f"Unsupported evaluator contract: {contract!r}")
    for name in ("get_model", "eval", "reset_model"):
        if not callable(getattr(adapter, name, None)):
            raise RuntimeError(f"Submitted ACT adapter lacks {name}")
    print(f"[SUBMISSION EVAL CONTRACT] {contract}", flush=True)
    return adapter


if __name__ == "__main__":
    base_eval.load_policy_adapter = load_submitted_adapter
    base_eval.main()
