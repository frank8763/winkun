"""Exact T272 auditor reused with the registered action-blind class and output."""
from blind_model import *
import audit as original_audit
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t274_action_identifiability'
DEST=ROOT/'stage/docs/long_distance_20261101/T274_AUDIT.json'


if __name__=='__main__':
    original_audit.HERE=HERE
    original_audit.OUT=OUT
    original_audit.DEST=DEST
    original_audit.EventModel=ActionBlind
    original_audit.main()
