"""Explicit bounded Gaussian-copula AR(1) policy and conditional likelihood.

The previous latent is recovered under CURRENT parameters at the fixed previous
state/action; its derivative is retained. No marginal-as-conditional shortcut.
"""
import math
import torch
SIGMA=(.035,.03)
RHO=.98
LOG2PI=math.log(2*math.pi)
def terms(mean,action):
    m=mean.double();a=action.double();s=torch.tensor(SIGMA,dtype=torch.float64,device=m.device)
    # Reflect positive/negative tails, evaluating the small Gaussian CDF via
    # log_ndtr, not 1-erf cancellation. The branch sign is locally constant.
    sign=torch.where(m>=0,torch.ones_like(m),-torch.ones_like(m));mm=sign*m;aa=sign*a
    lo=torch.special.log_ndtr((-1-mm)/s).exp();hi=torch.special.log_ndtr((1-mm)/s).exp();mass=hi-lo
    assert bool((mass>0).all()) and bool(torch.isfinite(mass).all())
    u=(torch.special.log_ndtr((aa-mm)/s).exp()-lo)/mass
    assert bool(((u>0)&(u<1)).all()),'Noninvertible rounded native action; reject rather than clamp density'
    z=sign*torch.special.ndtri(u)
    marginal=-.5*((a-m)/s).square()-s.log()-.5*LOG2PI-mass.log()
    assert bool(torch.isfinite(z).all()) and bool(torch.isfinite(marginal).all())
    return z,marginal
def sample(mean,latent,native=True):
    m=mean.double();s=torch.tensor(SIGMA,dtype=torch.float64,device=m.device);z=torch.as_tensor(latent,dtype=torch.float64,device=m.device)
    sign=torch.where(m>=0,torch.ones_like(m),-torch.ones_like(m));mm=sign*m
    lo=torch.special.log_ndtr((-1-mm)/s).exp();hi=torch.special.log_ndtr((1-mm)/s).exp();mass=hi-lo
    assert bool((mass>0).all())
    u=torch.special.log_ndtr(sign*z).exp();q=lo+u*mass
    assert bool(((q>0)&(q<1)).all())
    action=sign*(mm+s*torch.special.ndtri(q))
    if native:action=action.float()
    assert bool((action.abs()<1).all()) and bool(torch.isfinite(action).all())
    return action
def logp(mean,action,previous_mean,previous_action,reset,rho=RHO):
    z,marginal=terms(mean,action);prev,_=terms(previous_mean,previous_action)
    rr=torch.where(reset.bool(),torch.zeros_like(z[...,0]),torch.full_like(z[...,0],rho))[...,None]
    variance=1-rr.square()
    conditional=-.5*(z-rr*prev).square()/variance-.5*variance.log()-.5*LOG2PI
    standard=-.5*z.square()-.5*LOG2PI
    return (conditional+marginal-standard).sum(-1)
