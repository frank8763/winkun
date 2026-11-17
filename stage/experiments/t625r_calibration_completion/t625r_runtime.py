"""Only path recovery; reuse byte-identical T625 GPU fit/loss/evaluation code."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t625_target_continuation_calibration'))
import t625_train as T
M=T.M;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t625r_calibration_completion';DOC=ROOT/'stage/docs/day_20261117/t625r'
M.HERE=HERE;M.OUT=OUT;M.DOC=DOC
if __name__=='__main__':getattr(T,sys.argv[1])(sys.argv[2])
