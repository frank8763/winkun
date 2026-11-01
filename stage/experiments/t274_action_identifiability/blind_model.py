"""Capacity-matched T272 head with the two CURRENT action inputs masked.

Past real actions in the causal120 observation are retained. Only the queried
current two-dimensional action is removed, in training AND every evaluation.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t272_event_value_pilot'))
from event_model import *


class ActionBlind(EventModel):
    def __init__(self):
        super().__init__()
        mask=torch.ones(122);mask[-2:]=0;self.register_buffer('current_action_input_mask',mask)

    def forward(self,x):return super().forward(x*self.current_action_input_mask)
