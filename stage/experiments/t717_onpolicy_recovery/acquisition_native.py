"""Frozen neural recovery after exact learner prefix; diagnostics only."""
from pathlib import Path
import importlib.util,sys,os,gzip,json,time,fcntl
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('t717_recovery_controller',HERE/'run.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
N=C.M.N;Base=N.N;F=Base.F;A=C.A;np,torch=A.np,A.torch
NORMAL=Base.C.E.payload

class Session(Base.C.Session):
    def __init__(self,job,*args,**kw):
        super().__init__(job,*args,**kw);self.first=job['recovery_first'];self.source=Base.C.E.N.S.arrays(job['source_path']);assert C.sha(job['source_path'])==job['source_sha256'];self.prefix_actions=[];self.suffix_sample=bool(self.q.sample)
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0 and actor_active
        k=len(self.rows);prefix=k<self.first;self.q.sample=self.suffix_sample and not prefix
        action=self.source['actions'][k] if prefix else np.zeros(2,np.float32);self.prefix_actions.append(action.copy())
        return F.FieldSession.step(self,action if prefix else None,owner=int(prefix),actor_active=True)

def payload(session,tape=None):
    a=NORMAL(session,tape);a.update(source_prefix_actions=np.asarray(session.prefix_actions[:len(a['states'])],np.float32),recovery_first=np.asarray(session.first,np.int64));return a

def worker(pipe,index,p,counts):
    Base.C.ACTIVE=p;real=Base.G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;Base.G.T.Session=Session;Base.G.B.native_setup=lambda folder,pc:Base.C.C.consistent_reset.setup(F,folder,pc,Base.C.C.CANONICAL_SETUP)
    Base.P.frozen=A.frozen;Base.C.E.payload=payload;Base.G.T.worker(pipe,index,p,counts)

def inspect(p,a,row):
    job=row['request']['job'];k=int(a['recovery_first']);assert k==job['recovery_first']>=0 and len(a['states'])>k
    assert C.sha(job['source_path'])==job['source_sha256'];b=Base.C.E.N.S.arrays(job['source_path'])
    for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][:k],b[key][:k]),('prefix',key)
    for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],b[key][k]),('handoff',key)
    assert np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
    assert not a['stochastic'][:k].any() and (a['stochastic'][k:].all() if row['seed'] is not None else not a['stochastic'][k:].any())
    assert a['ownership'][:k].all() and not a['ownership'][k:].any()
    assert np.array_equal(a['source_prefix_actions'][:k],b['actions'][:k]) and not a['source_prefix_actions'][k:].any()
    proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32);assert np.array_equal(a['actions'][k:],proposals[k:])
    assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    Base.P.frozen=A.frozen
    F.M.audit_full_history(p,dict(row,teacher=False),dict(a,actions=proposals,ownership=np.zeros(len(a['states']),np.uint8)))
    assert all(A.digest(v)==row['array_hashes'][key] for key,v in a.items())
    assert int(a['native_counts'].sum())==row['physics']<=job['case']['step_cap']


def install():
    N.BASE_INSPECT=inspect
