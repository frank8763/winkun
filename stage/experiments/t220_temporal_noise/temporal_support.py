"""Frozen sensor-policy exploration diagnostic; no learning or controller override."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t219_rnd_runtime'))
import support as S
# Import this file as temporal_support, never as support: the historical learner
# and corrected native worker retain their unambiguous original import names.
C,R,np,torch=S.C,S.R,S.np,S.torch
OUT=ROOT/'stage/runs/t220_temporal_noise'
digest,state_np,used=S.digest,S.state_np,S.used
import os,json,time,math,traceback

def copula_action(mean,latent):
    mean=mean.double();sigma=torch.tensor(R.SIGMA,dtype=torch.float64)
    lo=torch.special.ndtr((-1-mean)/sigma);hi=torch.special.ndtr((1-mean)/sigma)
    u=torch.special.ndtr(torch.as_tensor(latent,dtype=torch.float64)).clamp(1e-12,1-1e-12)
    return (mean+sigma*torch.special.ndtri(lo+u*(hi-lo))).clamp(-1,1).float()

def worker(pipe,index,deadline):
    S.OUT=OUT;original_env=R.task_env;context={}
    def reset(create,job):
        context.clear();context['rho']=job['case']['noise_rho'];context['previous']=None
        return original_env(create,job)
    def draw(p,state,mode,rng):
        assert mode=='frozen_copula_diagnostic'
        rho=context['rho'];innovation=rng.normal(size=2)
        latent=innovation if context['previous'] is None else rho*context['previous']+math.sqrt(1-rho*rho)*innovation
        context['previous']=latent
        with torch.no_grad():mean=p.physical_mean(torch.tensor(state[None])).double()[0];action=copula_action(mean,latent)
        # Zero placeholder, NOT a behavior log density. No PPO/replay learning
        # may consume these temporally correlated diagnostic records.
        return action.numpy(),mean.float().numpy(),0.
    R.task_env=reset;R.draw=draw
    S.worker(pipe,index,deadline)
