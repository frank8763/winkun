"""Direct narrow-task transfer with a declared current-policy support anchor."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t353_offline_night_chain'))
from night_core import *
import night_core as NC
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/narrow_20261105';OUT=ROOT/'stage/runs/t358_direct_narrow'


def references(p,q):
    old,induced,supports=NC.references(p,q)
    for g in supports:
        with torch.no_grad():g['y']=q.sequence(g['x'],g['b']).detach().clone()
        assert float((q.sequence(g['x'],g['b']).detach()-g['y']).abs().max())==0.
    return old,induced,supports
