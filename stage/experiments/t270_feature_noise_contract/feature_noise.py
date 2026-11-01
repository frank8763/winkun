"""Bounded feature-correlated exploration with exact Gaussian latent filtering.

Frozen unit-norm causal features; one latent matrix per50 decisions and a small
independent residual. Known history is assimilated under CURRENT actor means.
This is a standard latent-Gaussian/control-density construction, not a novelty claim.
"""
from pathlib import Path
import math
import sys
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t238_ar_density_recovery'))
import copula as AR

BLOCK=50
FEATURE_DIM=32
LATENT_VAR=.98
RESIDUAL_VAR=.02


def unit_feature(hidden):
    feature=hidden.reshape(-1).double()
    assert feature.shape==(FEATURE_DIM,)
    norm=feature.norm();assert float(norm)>1e-8
    return feature/norm


def prior(feature,latent_var=LATENT_VAR):
    return feature.new_zeros((len(feature),2)),torch.eye(len(feature),dtype=feature.dtype,device=feature.device)*latent_var


def logp_step(mean,action,feature,m,P,residual_var=RESIDUAL_VAR):
    """Returns density and posterior. Do not detach m: it contains past means."""
    z,marginal=AR.terms(mean,action)
    v=feature@P@feature+residual_var
    assert float(v)>0 and bool(torch.isfinite(v))
    location=feature@m
    q=-.5*((z-location).square()/v+v.log()+AR.LOG2PI)
    standard=-.5*z.square()-.5*AR.LOG2PI
    lp=(q+marginal-standard).sum()
    gain=P@feature/v
    new_m=m+gain[:,None]*(z-location)[None,:]
    new_P=P-torch.outer(P@feature,P@feature)/v
    new_P=(new_P+new_P.T)/2
    return lp,new_m,new_P


def sequence_logp(means,actions,features,block=BLOCK,latent_var=LATENT_VAR,residual_var=RESIDUAL_VAR):
    assert len(means)==len(actions)==len(features)
    lp=[]
    for k,(mean,a,f) in enumerate(zip(means,actions,features)):
        if k%block==0:m,P=prior(f,latent_var)
        value,m,P=logp_step(mean,a,f,m,P,residual_var)
        lp.append(value)
    return torch.stack(lp)


def latent_sample(feature,weights,innovation):
    return feature@weights+math.sqrt(RESIDUAL_VAR)*torch.as_tensor(innovation,dtype=torch.float64)
