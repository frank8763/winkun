"""Same T239 implementation; only the registered q99 guard changes."""
from pathlib import Path
import sys,importlib.util
_ROOT=Path(__file__).resolve().parents[3]
_spec=importlib.util.spec_from_file_location('stage239_arbase',_ROOT/'stage/experiments/t239_ar_policy_gradient/ar_support.py')
_A=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_A;_spec.loader.exec_module(_A)
from stage239_arbase import *
ROOT=_ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t240_retention_ablation'
def worker(pipe,index,deadline):
    _A.OUT=OUT
    _A.worker(pipe,index,deadline)
