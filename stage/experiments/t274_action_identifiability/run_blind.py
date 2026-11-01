"""Exactly T272's 500-step schedule with only current-action input disabled."""
from blind_model import *
import train as registered_training
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t274_action_identifiability'


if __name__=='__main__':
    # The full original registered trainer is source-hash locked, not edited.
    # Its loss, data normalization, episode/row draws, Adam and gates are reused.
    registered_training.HERE=HERE
    registered_training.OUT=OUT
    registered_training.EventModel=ActionBlind
    registered_training.main()
