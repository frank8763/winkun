"""Frozen Stage parent/candidate exploration, no learning or action scripts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t232_complementary_sources'
sys.path.insert(0,str(ROOT/'stage/experiments/t223_coverage_repair'))
import repair_support as B
S,C,R,np,torch=B.S,B.C,B.R,B.np,B.torch
L=B.L
import os,json,time,copy,traceback
used,digest,state_np=S.used,S.digest,S.state_np

def worker(pipe,index,deadline):
    B.OUT=OUT
    R.SIGMA=(.035,.03)
    original=B.TS.copula_action
    def checked(mean,latent):
        sigma=B.torch.tensor(R.SIGMA,dtype=B.torch.float64)
        lo=B.torch.special.ndtr((-1-mean.double())/sigma);hi=B.torch.special.ndtr((1-mean.double())/sigma)
        assert bool((hi>lo).all()),'TruncatedGaussianCDFunderflow: no silently reversed action'
        result=original(mean,latent);assert bool(B.torch.isfinite(result).all()) and bool((result.abs()<=1).all())
        return result
    B.TS.copula_action=checked
    B.worker(pipe,index,deadline)
