"""Expose the participant package to an unchanged official evaluator."""
import sys, runpy, os
from pathlib import Path
root=Path(sys.argv[1]).resolve()
os.chdir(root)
sys.path.insert(0,str(root))
import policy
policy.__path__=[str(Path(__file__).resolve().parent.parent),*policy.__path__]
sys.argv=[str(root/'scripts/eval_policy.py'),*sys.argv[2:]]
runpy.run_path(sys.argv[0],run_name='__main__')
