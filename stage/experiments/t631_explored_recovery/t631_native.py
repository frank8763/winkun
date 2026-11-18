"""Fixed real prefix, seeded exploration only after handoff; no teacher labels."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t630_early_departure_recovery'))
import t630_native as Old
P=Old.Prior
N=P.N


class Session(P.Session):
    def __init__(self,job,*args,**kwargs):
        super().__init__(job,*args,**kwargs)
        self.explore=bool(job.get('explore_after_prefix',False))

    def step(self,forced=None,owner=0,actor_active=True):
        assert forced is None and owner==0
        if self.first<0:return N.Session.step(self)
        k=len(self.rows);prefix=k<self.first
        self.q.sample=self.explore and not prefix
        assert not self.q.sample or self.q.rng is not None
        u=self.source['actions'][k] if prefix else N.np.zeros(2,N.np.float32)
        self.prefix_actions.append(u.copy())
        return N.F.FieldSession.step(self,u if prefix else None,
                                     owner=int(prefix),actor_active=True)


def inspect(p,values):
    for a,r in values:
        if 'student_takeover_first' not in a:
            P.inspect(p,[(a,r)]);continue
        job=r['request']['job'];b=N.E.N.S.arrays(job['source_path'])
        k=int(a['student_takeover_first']);assert k==job['student_takeover_first']
        assert 0<k<len(a['states'])
        for key in ['states','next_states','actions','native_counts','poses','next_poses',
                    'history_poses','history_goals','history_times']:
            assert N.np.array_equal(a[key][:k],b[key][:k]),key
        for key in ['states','poses','history_poses','history_goals','history_times']:
            assert N.np.array_equal(a[key][k],b[key][k]),key
        assert N.np.array_equal(a['initial_native_integration'],b['initial_native_integration'])
        assert a['ownership'][:k].all() and not a['ownership'][k:].any()
        assert not a['stochastic'][:k].any()
        expected=bool(job.get('explore_after_prefix',False))
        assert N.np.all(a['stochastic'][k:]==expected)
        assert not expected or r['seed'] is not None
        assert N.np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
        assert N.np.array_equal(a['source_prefix_actions'][:k],b['actions'][:k])
        assert not a['source_prefix_actions'][k:].any()
        commands=N.np.clip(a['raw_actions'],-1,1).astype(N.np.float32)
        assert N.np.array_equal(commands[k:],a['actions'][k:])
        # Check all own network proposals/hidden/RNG/density/reward. Forced
        # prefix actions remain separately checked against the physical source.
        # The original handoff dispatcher handles deterministic-only suffixes.
        # Prefix assertions above replace that dispatcher; the general full
        # history audit below still reconstructs every network/RNG/reward row.
        own={key:v for key,v in a.items() if key not in ['student_takeover_first','source_prefix_actions']}
        own.update(actions=commands,ownership=N.np.zeros(len(commands),N.np.uint8))
        P.BASE_INSPECT(p,[(own,dict(r,teacher=False))])


def worker(pipe,index,p,counts):
    os.environ['CUDA_VISIBLE_DEVICES']='';assert not N.torch.cuda.is_initialized()
    P.install();real=N.G._real_affinity
    def pin(pid,mask):
        return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;N.G.T.Session=Session
    N.G.B.native_setup=lambda folder,plan:Old.reset_contract.field_setup(N.F,folder,plan)
    N.E.payload=P.payload;N.G.T.worker(pipe,index,p,counts)
