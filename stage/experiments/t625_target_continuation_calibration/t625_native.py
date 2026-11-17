"""Original T623 controls; corrected contract and actual three-core mapping."""
from pathlib import Path
import sys,importlib.util
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t624_exact_branch_completion'))
import t624_native as V
N=V.N;worker=V.worker;Pool=N.Pool
def inspect(p,values):
 spec=importlib.util.spec_from_file_location('t625_unique_recovery_contract',ROOT/'stage/experiments/t623r_branch_contract_recovery/recover.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 source=N.E.N.S.arrays(p['prefix_source'])
 for a,r in values:m.check(p,a,r,source)
