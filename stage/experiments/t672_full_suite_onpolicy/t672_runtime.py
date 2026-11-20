"""Unmodified canonical current-policy sampler, finite per-batch children."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t672_full_suite_onpolicy';DOC=ROOT/'stage/docs/weekend_20261120_23/t672'
import t672_wait as Wait
sys.path.insert(0,str(ROOT/'stage/experiments/t665_remaining_verification'))
import t665_runtime as T
T.OUT,T.DOC,T.HERE,T.Wait=OUT,DOC,HERE,Wait
if __name__=='__main__':{'collect':T.frozen_main,'audit':T.frozen_audit}[sys.argv[1]]()
