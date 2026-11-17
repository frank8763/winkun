"""Use the original T623 control path with actual CPU-affinity restoration."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t623_common_continuation_branches'))
import t623_native as N
ORIGINAL_WORKER=N.worker
def worker(pipe,index,p,counts):
 # Bypass the old T493 import wrapper that remaps all new CPU IDs to14/15.
 os.sched_setaffinity=N.G._real_affinity
 ORIGINAL_WORKER(pipe,index,p,counts)
def restore_parent(p):
 real=N.G._real_affinity;real(0,set(p['stage_cpus']));assert set(os.sched_getaffinity(0))==set(p['stage_cpus'])
 os.sched_setaffinity=real
