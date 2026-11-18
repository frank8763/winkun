"""Normal autonomous evaluation or genuine stochastic actual-state recovery."""
from pathlib import Path
import os,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t644_sampled_recovery'))
import t644_native as Prior
N,G,F,P,E,L,np,torch=Prior.N,Prior.G,Prior.F,Prior.P,Prior.E,Prior.L,Prior.np,Prior.torch


class Session(F.FieldSession):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        self.first=int(job.get('student_takeover_first',-1))
        self.source=E.N.S.arrays(job['source_path']) if self.first>=0 else None
        self.prefix_actions=[]

    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        if self.first<0:return super().step()
        k=len(self.rows);use=k<self.first;self.q.sample=not use
        assert self.q.rng is not None
        u=self.source['actions'][k] if use else np.zeros(2,np.float32);self.prefix_actions.append(u.copy())
        return super().step(u if use else None,owner=int(use),actor_active=True)


def payload(s,tape=None):
    a=Prior.Prior.NORMAL(s,tape)
    if s.first>=0:a.update(source_prefix_actions=np.asarray(s.prefix_actions[:len(a['states'])],np.float32),
                         student_takeover_first=np.asarray(s.first,np.int64))
    return a


def worker(pipe,index,p,counts):
    real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Session
    G.B.native_setup=lambda folder,pc:N.Reset.field_setup(F,folder,pc)
    P.frozen=L.frozen;E.payload=payload;G.T.worker(pipe,index,p,counts)


def inspect(p,a,row):
    if 'student_takeover_first' not in a:
        assert not a['ownership'].any() and not a['stochastic'].any()
        F.M.audit_full_history(p,row,a)
    else:
        pc=dict(p,first_divergence=dict(decision=int(a['student_takeover_first'])))
        Prior.inspect(pc,a,row)
    assert all(N.D.digest(v)==row['array_hashes'][k] for k,v in a.items())
