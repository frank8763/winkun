"""Frozen Stage parent/candidate exploration, no learning or action scripts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t226_local_exploration'
sys.path.insert(0,str(ROOT/'stage/experiments/t223_coverage_repair'))
import repair_support as B
S,C,R,np,torch=B.S,B.C,B.R,B.np,B.torch
L=B.L
import os,json,time,copy,traceback
used,digest,state_np=S.used,S.digest,S.state_np

def worker(pipe,index,deadline):
    B.OUT=OUT
    R.SIGMA=(.035,.03)  # Only native exploration scale differs from T225 controls.
    B.worker(pipe,index,deadline)
