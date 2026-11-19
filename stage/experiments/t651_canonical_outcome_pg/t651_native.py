"""Guard exact canonical scenes, goals and available initial120 before stepping."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t650_reset_contract_repair'))
import t650_native as C
G,F,E,P,L,np,torch=C.G,C.F,C.E,C.P,C.L,C.np,C.torch
ACTIVE=None

class Session(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        expected=ACTIVE['narrow_scene'] if 'fullfield_start_pose' in job or not job.get('wide_course') else ACTIVE['wide_scene']
        assert self.scene==expected,'Task static geometry changed before first action'
        if job.get('radial_distance_m')==8:
            assert np.array_equal(self.goal,np.asarray(ACTIVE['radial8_goals'][job['case']['side']])),'Eight-metre goal changed'
        if 'initial_contract' in job:
            v=job['initial_contract'];assert self.scene==v['scene_sha256']
            assert E.N.K.R.arrays_hash(self.initial)==v['integration_sha256']
            assert E.N.K.R.arrays_hash(self.packet())==v['initial120_sha256']
            assert np.array_equal(self.goal,np.asarray(v['goal_xy']))
        assert self.steps==0 and not self.rows and not self.hidden.any()

def worker(pipe,index,p,counts):
    global ACTIVE
    ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:C.consistent_reset.setup(F,folder,pc,C.CANONICAL_SETUP)
    P.frozen=L.frozen;G.T.worker(pipe,index,p,counts)
