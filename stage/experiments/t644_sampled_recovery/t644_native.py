"""Independent fixed Gaussian innovations after a genuine candidate prefix."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t643_old_policy_recovery'))
import t643_native as Prior
N,G,F,P,E,L,np,torch=Prior.N,Prior.G,Prior.F,Prior.P,Prior.E,Prior.L,Prior.np,Prior.torch


class Session(Prior.Session):
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        k=len(self.rows);use=k<self.first
        self.q.sample=not use;assert self.q.rng is not None
        u=self.source['actions'][k] if use else np.zeros(2,np.float32);self.prefix_actions.append(u.copy())
        return F.FieldSession.step(self,u if use else None,owner=int(use),actor_active=True)


def worker(pipe,index,p,counts):
    real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:N.Reset.field_setup(F,folder,pc)
    P.frozen=L.frozen;E.payload=Prior.payload;G.T.worker(pipe,index,p,counts)


def inspect(p,a,row):
    k=int(a['student_takeover_first']);assert k==p['first_divergence']['decision']>0
    b=E.N.S.arrays(row['request']['job']['source_path']);assert len(a['states'])>k
    for name in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:
        assert np.array_equal(a[name][:k],b[name][:k]),name
    for name in ['states','poses','history_poses','history_goals','history_times']:
        assert np.array_equal(a[name][k],b[name][k]),name
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
    assert not a['stochastic'][:k].any() and a['stochastic'][k:].all()
    assert a['ownership'][:k].all() and not a['ownership'][k:].any()
    assert np.array_equal(a['source_prefix_actions'][:k],b['actions'][:k]) and not a['source_prefix_actions'][k:].any()
    proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32)
    assert np.array_equal(a['actions'][k:],proposals[k:])
    assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    assert row['seed'] is not None
    F.M.audit_full_history(p,dict(row,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
    assert all(N.D.digest(v)==row['array_hashes'][name] for name,v in a.items())
