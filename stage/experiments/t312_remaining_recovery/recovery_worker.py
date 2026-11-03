from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t312_online_goal_ppo'))
import ppo_worker as original

def worker(pipe,index,p,counts):
    original.OUT=ROOT/'stage/runs/t312_remaining_recovery'
    original.B.OUT=original.OUT
    return original.worker(pipe,index,p,counts)
