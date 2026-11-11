"""Exact T488 learning function objects; budget/data orchestration recovery only."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t488_left_failure_focus'))
import t488_learning as F
C,P,E,M,R,torch,np=F.C,F.P,F.E,F.M,F.R,F.torch,F.np
read,save,sha,digest=F.read,F.save,F.sha,F.digest
make,fit=F.make,F.fit
