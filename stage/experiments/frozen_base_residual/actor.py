"""Preparation-only frozen Stage base plus trainable observation residual.

Not a trained policy or a registered native experiment. No rollout, optimizer,
external controller, expert dispatch, distance router or recurrent carry.
"""
import torch


class FrozenBaseResidual(torch.nn.Module):
    def __init__(self, frozen, center, mode, mean, scale, seed=2026110324):
        super().__init__()
        assert not frozen.memory_on
        self.frozen = frozen.eval().requires_grad_(False)
        self.center_fn, self.mode_fn = center, mode
        self.memory_on = False
        assert mean.shape == scale.shape == (120,)
        assert torch.isfinite(mean).all() and torch.isfinite(scale).all()
        assert (scale > 0).all()
        self.register_buffer('input_mean', mean.detach().clone())
        self.register_buffer('input_scale', scale.detach().clone())
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.correction = torch.nn.Sequential(
                torch.nn.Linear(120, 64), torch.nn.Tanh(), torch.nn.Linear(64, 2))
            torch.nn.init.zeros_(self.correction[-1].weight)
            torch.nn.init.zeros_(self.correction[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    def step(self, x, hidden=None):
        assert x.ndim == 2 and x.shape[-1] == 120
        # Keep the exact original raw -> bounded mode / stochastic-center paths.
        # The center is NOT atanh(mode) in this Stage implementation.
        with torch.no_grad():
            _, _, new, base_raw = self.frozen.step(x, None)
        correction = self.correction(
            ((x-self.input_mean)/self.input_scale).clamp(-10., 10.))
        raw = base_raw + correction
        return self.mode_fn(raw), self.center_fn(raw), new, raw

    def sequence(self, x):
        assert x.ndim == 3 and x.shape[-1] == 120
        shape = x.shape[:2]
        rows = [self.step(row[None]) for row in x.reshape(-1, 120)]
        return tuple(torch.cat([r[k] for r in rows]).reshape(*shape, 2)
                     for k in (0, 1, 3))
