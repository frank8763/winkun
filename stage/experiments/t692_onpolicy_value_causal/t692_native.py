"""Canonical actual prefix, one forced command, then own stochastic policy."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t651_canonical_outcome_pg'))
import t651_native as C
G,F,E,P,L,np,torch=C.G,C.F,C.E,C.P,C.L,C.np,C.torch
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
import t654_native as Legacy
NORMAL=Legacy.NORMAL

class Session(C.Session):
    def __init__(self,job,*args,**kw):
        super().__init__(job,*args,**kw)
        assert L.sha(job['source_path'])==job['source_sha256']
        self.source=E.N.S.arrays(job['source_path']);self.branch=int(job['branch_decision']);assert self.branch==20
        self.branch_command=np.clip(self.source['locations'][self.branch]+np.asarray(job['action_delta']),-1,1).astype(np.float32)
        assert np.array_equal(self.initial,self.source['initial_native_integration']) and np.array_equal(self.goal,self.source['task_goal_xy'])
    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0 and actor_active
        i=len(self.rows)
        if i<=self.branch:assert np.array_equal(self.packet(),self.source['states'][i]),('prefix_packet',i)
        self.q.sample=i>self.branch;assert self.q.rng is not None
        command=self.source['actions'][i] if i<self.branch else (self.branch_command if i==self.branch else None)
        return F.FieldSession.step(self,command,owner=int(i<=self.branch),actor_active=True)

def payload(session,tape=None):return NORMAL(session,tape)

def worker(pipe,index,p,counts):
    C.ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:C.C.consistent_reset.setup(F,folder,pc,C.C.CANONICAL_SETUP)
    P.frozen=L.frozen;E.payload=payload;G.T.worker(pipe,index,p,counts)

def inspect(p,a,row):
    j=row['request']['job'];k=j['branch_decision'];source=E.N.S.arrays(j['source_path']);n=len(a['states']);assert n>k
    mask=(np.arange(n)<=k).astype(np.uint8)
    assert np.array_equal(a['ownership'],mask) and np.array_equal(a['stochastic'],1-mask)
    for name in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:
        assert np.array_equal(a[name][:k],source[name][:k]),name
    for name in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[name][k],source[name][k]),name
    assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
    branch=np.clip(source['locations'][k]+np.asarray(j['action_delta']),-1,1).astype(np.float32);assert np.array_equal(branch,a['actions'][k])
    assert np.array_equal(a['states'][1:],a['next_states'][:-1]) and np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    proposals=np.clip(a['raw_actions'],-1,1).astype(np.float32)
    assert np.array_equal(a['actions'][k+1:],proposals[k+1:]) and not a['innovations'][:k+1].any()
    # This independently reconstructs model tape/own carry, current stochastic
    # flags and RNG, recorded history geometry and rewards. Only the forced
    # execution columns are checked against their explicit intervention above.
    F.M.audit_full_history(p,dict(row,teacher=False),dict(a,actions=proposals,ownership=np.zeros(n,np.uint8)))
    expected=np.random.default_rng(np.random.SeedSequence(row['seed'])).normal(size=a['innovations'][k+1:].shape)
    assert np.array_equal(expected,a['innovations'][k+1:]) and int(a['native_counts'].sum())==row['physics']
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
    assert all(L.digest(v)==row['array_hashes'][name] for name,v in a.items())
