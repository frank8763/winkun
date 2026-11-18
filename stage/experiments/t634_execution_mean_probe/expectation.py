"""Exact coordinate-wise expectation of clip(N(mu,sigma),-1,1).

This is a fixed deterministic output transform, not E[nonlinear dynamics] or
an expected return. It has no planner, teacher, sampling or fitted parameter.
"""
import math
import numpy as np


def clipped_mean(mu,sigma):
    mu=np.asarray(mu,dtype=np.float64);sigma=np.asarray(sigma,dtype=np.float64)
    assert mu.shape[-1]==2 and sigma.shape==(2,)
    assert np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma>0).all()
    lo=(-1-mu)/sigma;hi=(1-mu)/sigma
    cdf=lambda x:np.asarray([.5*math.erfc(-float(v)/math.sqrt(2)) for v in x.flat]).reshape(x.shape)
    phi=lambda x:np.exp(-.5*x*x)/math.sqrt(2*math.pi)
    value=-cdf(lo)+mu*(cdf(hi)-cdf(lo))+sigma*(phi(lo)-phi(hi))+(1-cdf(hi))
    assert np.isfinite(value).all() and (np.abs(value)<=1+1e-12).all()
    return np.clip(value,-1,1)
