"""Reuse verified own-history teacher-prefix IO, corrected reset assertion."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t629_offline_night_relay'))
import t629_native as Prior
import reset_contract


def worker(pipe,index,p,counts):
    N=Prior.N
    os.environ['CUDA_VISIBLE_DEVICES']=''
    assert not N.torch.cuda.is_initialized()
    Prior.install()
    real=N.G._real_affinity
    def pin(pid,mask):
        return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin
    N.G.T.Session=Prior.Session
    N.G.B.native_setup=lambda folder,plan:reset_contract.field_setup(N.F,folder,plan)
    N.E.payload=Prior.payload
    N.G.T.worker(pipe,index,p,counts)
