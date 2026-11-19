"""Frozen T501 canonical reset; strong contract before the first physical step."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t645_iterative_recovery_relay'))
import t645_native as Prior
import consistent_reset
G,F,E,P,L,np,torch=Prior.G,Prior.F,Prior.E,Prior.P,Prior.L,Prior.np,Prior.torch
CANONICAL_SETUP=G.B.native_setup


class Session(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        expected=job['initial_contract']
        assert self.scene==expected['scene_sha256'],'Initial static geometry differs from canonical reference'
        assert E.N.K.R.arrays_hash(self.initial)==expected['integration_sha256']
        packet=self.packet()
        assert E.N.K.R.arrays_hash(packet)==expected['initial120_sha256'],'Initial120 differs before any action'
        assert np.array_equal(self.goal,np.asarray(expected['goal_xy']))
        assert not self.hidden.any() and self.steps==0 and not self.rows


def worker(pipe,index,p,counts):
    real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:consistent_reset.setup(F,folder,pc,CANONICAL_SETUP)
    P.frozen=L.frozen;G.T.worker(pipe,index,p,counts)
