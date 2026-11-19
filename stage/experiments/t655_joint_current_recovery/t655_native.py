"""Fixed current-policy prefix plus fresh student suffix; canonical dynamics."""
from pathlib import Path
import sys,os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
import t654_native as B
G,F,E,P,L,np,torch=B.G,B.F,B.E,B.P,B.L,B.np,B.torch
Prior=B.Prior

class Session(Prior.Session):
    def __init__(self,job,*args,**kw):
        super().__init__(job,*args,**kw)
        self.first=int(job.get('recovery_first',-1));self.source=E.N.S.arrays(job['source_path']) if self.first>=0 else None;self.prefix_actions=[]
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        if self.first<0:return super().step()
        k=len(self.rows);use=k<self.first;self.q.sample=not use;assert self.q.rng is not None
        u=self.source['actions'][k] if use else np.zeros(2,np.float32);self.prefix_actions.append(u.copy())
        return F.FieldSession.step(self,u if use else None,owner=int(use),actor_active=True)

def payload(s,tape=None):return B.payload(s,tape)

def worker(pipe,index,p,counts):
    Prior.ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:Prior.C.consistent_reset.setup(F,folder,pc,Prior.C.CANONICAL_SETUP)
    P.frozen=L.frozen;E.payload=payload;G.T.worker(pipe,index,p,counts)

def inspect(p,a,row):
    k=int(a.get('recovery_first',0));job=row['request']['job']
    if 'recovery_first' in a:
        assert k==job['recovery_first']>0
        b=E.N.S.arrays(job['source_path']);assert len(a['states'])>k
        for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:
            assert np.array_equal(a[key][:k],b[key][:k]),key
        for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],b[key][k]),key
        assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
        assert a['ownership'][:k].all() and not a['ownership'][k:].any()
        assert not a['stochastic'][:k].any() and a['stochastic'][k:].all()
        assert np.array_equal(a['source_prefix_actions'][:k],b['actions'][:k]) and not a['source_prefix_actions'][k:].any()
    else:assert not a['ownership'].any()
    proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32)
    assert np.array_equal(a['actions'][k:],proposals[k:])
    assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    F.M.audit_full_history(p,dict(row,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
    assert all(L.digest(v)==row['array_hashes'][key] for key,v in a.items())
