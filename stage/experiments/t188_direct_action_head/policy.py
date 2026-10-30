"""A zero-initialized direct feature-to-native-command branch in one Actor."""
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t177_frontier_onpolicy'))
import common as C

class DirectFeaturePolicy(C.Policy):
    def __init__(self):
        super().__init__()
        self.direct_head=torch.nn.Linear(600,2)
        torch.nn.init.zeros_(self.direct_head.weight)
        torch.nn.init.zeros_(self.direct_head.bias)

    def forward(self,state):
        features=self.body.net[:4](state)
        old_action=self.body.net[4:](features)
        return self.native_head(old_action)+self.direct_head(features)

    def load_parent(self,state):
        result=self.load_state_dict(state,strict=False)
        assert set(result.missing_keys)=={'direct_head.weight','direct_head.bias'}
        assert not result.unexpected_keys
        assert torch.count_nonzero(self.direct_head.weight)==0
        assert torch.count_nonzero(self.direct_head.bias)==0
