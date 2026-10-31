"""Causal, fully trainable neural Actor candidate; no planner or runtime teacher.

Architecture only. Online collection/training is deliberately not implemented here.
The zero-initialized residual preserves the T194 function for initial comparisons.
"""
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t203_native_exploration'))
from runtime import C, center, log_density

BASE_DIM = 113
PACKET_DIM = 118
HIDDEN = 96


def packet(observation113, previous_action, steering_angles_rad, previous_dt_s):
    """Only current observation, previous action/time and measured steering.

    Added steering values are simulation encoder measurements; real deployment
    would need measured steering. Never supply future actions/labels or geometry.
    """
    if observation113.shape[-1] != BASE_DIM:
        raise ValueError('Expected existing 113 causal observation features')
    if previous_action.shape[-1] != 2 or steering_angles_rad.shape[-1] != 2:
        raise ValueError('Two native actions and two measured steering angles required')
    dt = previous_dt_s.unsqueeze(-1) / .2
    result = torch.cat((observation113, previous_action, steering_angles_rad, dt), -1)
    if result.shape[-1] != PACKET_DIM or not bool(torch.isfinite(result).all()):
        raise ValueError('Non-finite or malformed causal packet')
    if bool((previous_dt_s < 0).any()):
        raise ValueError('Elapsed time cannot be negative')
    return result


class SensorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rays = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.SiLU(),
            nn.Conv1d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(6), nn.Flatten(), nn.Linear(192, 64), nn.SiLU())
        self.context = nn.Sequential(nn.Linear(PACKET_DIM - 48, 64), nn.SiLU())

    def forward(self, x):
        return torch.cat((self.rays(x[:, :48].unsqueeze(1)), self.context(x[:, 48:])), -1)


class TemporalCore(nn.Module):
    def __init__(self, recurrent=True):
        super().__init__()
        self.encoder = SensorEncoder()
        self.cell = nn.GRUCell(128, HIDDEN)
        self.recurrent = recurrent

    def step(self, x, hidden, reset_before):
        if hidden is None:
            hidden = x.new_zeros((len(x), HIDDEN))
        hidden = torch.where(reset_before[:, None], torch.zeros_like(hidden), hidden)
        if not self.recurrent:
            hidden = torch.zeros_like(hidden)
        return self.cell(self.encoder(x), hidden)


class TemporalActor(nn.Module):
    def __init__(self, parent_state, recurrent=True):
        super().__init__()
        self.base = C.OriginalPolicy()
        self.base.load_state_dict(parent_state)
        self.core = TemporalCore(recurrent=recurrent)
        self.residual = nn.Linear(HIDDEN, 2)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        self.requires_grad_(True)
        # This candidate retains the T203 native-domain law with fixed sigma.
        self.base.logstd.requires_grad_(False)

    @staticmethod
    def mode(raw):
        return torch.tanh(torch.atanh(raw.clamp(-.999999, .999999)))

    def raw_base(self, x):
        return C.ParentPolicy.forward(self.base, x[..., :BASE_DIM])

    def step(self, x, hidden=None, reset_before=None):
        if x.ndim != 2 or x.shape[-1] != PACKET_DIM:
            raise ValueError('Actor step expects batch x 118')
        if reset_before is None:
            reset_before = torch.zeros(len(x), dtype=torch.bool, device=x.device)
        hidden = self.core.step(x, hidden, reset_before)
        raw = self.raw_base(x) + self.residual(hidden)
        return {'mode': self.mode(raw), 'native_mean': center(raw), 'raw': raw, 'hidden': hidden}

    def sequence(self, x, reset_before, hidden=None):
        if x.ndim != 3 or x.shape[-1] != PACKET_DIM or reset_before.shape != x.shape[:2]:
            raise ValueError('Sequence expects time x batch x118 and matching reset mask')
        raw_base = self.raw_base(x.reshape(-1, PACKET_DIM)).reshape(*x.shape[:2], 2)
        modes, means, raws, hidden_rows = [], [], [], []
        for t in range(len(x)):
            hidden = self.core.step(x[t], hidden, reset_before[t])
            raw = raw_base[t] + self.residual(hidden)
            modes.append(self.mode(raw)); means.append(center(raw)); raws.append(raw); hidden_rows.append(hidden)
        return {'mode': torch.stack(modes), 'native_mean': torch.stack(means),
                'raw': torch.stack(raws), 'hidden_sequence': torch.stack(hidden_rows), 'hidden': hidden}

    def log_prob(self, result, action):
        return log_density(result['native_mean'], action)


class TemporalValue(nn.Module):
    """New independent observation value baseline; no old Q weights/gradient."""
    def __init__(self, recurrent=True):
        super().__init__()
        self.core = TemporalCore(recurrent=recurrent)
        self.value = nn.Linear(HIDDEN, 1)

    def sequence(self, x, reset_before, hidden=None):
        values = []
        for t in range(len(x)):
            hidden = self.core.step(x[t], hidden, reset_before[t])
            values.append(self.value(hidden).squeeze(-1))
        return torch.stack(values), hidden


class PredictionHeads(nn.Module):
    """Optional future-training auxiliary only; disabled in Actor forward.

    1/3s goal-distance-change targets and contact hazard require causal temporal
    alignment, censor masks and held-out whole-episode validation before use.
    No safety filtering or action replacement is implemented.
    """
    def __init__(self):
        super().__init__()
        self.progress = nn.Linear(HIDDEN, 2)
        self.contact_logits = nn.Linear(HIDDEN, 2)

    def forward(self, hidden):
        return {'progress_1s_3s': self.progress(hidden), 'contact_logits_1s_3s': self.contact_logits(hidden)}
