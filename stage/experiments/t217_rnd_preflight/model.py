"""Training-only LiDAR RND; never produces or replaces navigation actions."""
import torch
from torch import nn


class LidarRND(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('input_mean', mean.clone())
        self.register_buffer('input_std', std.clamp_min(.05).clone())
        def net():
            return nn.Sequential(nn.Linear(48,128),nn.LeakyReLU(.1),nn.Linear(128,128),nn.LeakyReLU(.1),nn.Linear(128,64))
        self.target=net().requires_grad_(False)
        self.predictor=net()

    def features(self, rays):
        if rays.shape[-1]!=48 or not bool(torch.isfinite(rays).all()):
            raise ValueError('48 finite normalized LiDAR ranges required')
        if bool(((rays<0)|(rays>1)).any()):
            raise ValueError('LiDAR normalized range outside [0,1]')
        return ((rays-self.input_mean)/self.input_std).clamp(-5,5)

    def error(self,rays):
        x=self.features(rays)
        with torch.no_grad():target=self.target(x)
        return (self.predictor(x)-target).square().mean(-1)

    @torch.no_grad()
    def bonus(self,next_rays,native_counts,collision,physics_cap,scale,total_cap=.05):
        """Post-step training reward only, scored before predictor is updated.

        For one original episode with sum(counts)<=physics_cap, raw and physically
        discounted sums cannot exceed total_cap. No goal/distance/map features.
        Collision transitions receive zero intrinsic reward. True partial rollouts
        must be handled by the future collector, not relabeled as full episodes.
        """
        if physics_cap<=0 or scale<=0 or total_cap<0:
            raise ValueError('Positive episode scale/cap required')
        if next_rays.shape[:-1]!=native_counts.shape or collision.shape!=native_counts.shape:
            raise ValueError('Reward/transition shapes differ')
        if bool(((native_counts<=0)|(native_counts>100)).any()) or int(native_counts.sum())>physics_cap:
            raise ValueError('Native counts exceed the single-episode contract')
        raw=self.error(next_rays).div(scale).clamp(0,1)
        return total_cap*native_counts/physics_cap*raw*(~collision.bool())
