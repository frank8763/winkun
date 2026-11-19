"""Original-start current student only, canonical scene and causal carry."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t651_canonical_outcome_pg'))
import t651_native as Base
G,F,E,P,L,np,torch=Base.G,Base.F,Base.E,Base.P,Base.L,Base.np,Base.torch

def worker(pipe,index,p,counts):
    Base.ACTIVE=p;real=G._real_affinity
    def pin(pid,mask):return real(pid,{p['stage_cpus'][index]}) if pid==0 and set(mask)=={index} else real(pid,mask)
    os.sched_setaffinity=pin;G.T.Session=Base.Session
    G.B.native_setup=lambda folder,pc:Base.C.consistent_reset.setup(F,folder,pc,Base.C.CANONICAL_SETUP)
    P.frozen=L.frozen;G.T.worker(pipe,index,p,counts)

def inspect(p,a,row):
    assert not a['ownership'].any()
    assert bool(a['stochastic'].all()) if row['seed'] is not None else not a['stochastic'].any()
    assert np.array_equal(a['actions'],np.clip(a['raw_actions'],-1,1).astype(np.float32))
    assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
    F.M.audit_full_history(p,row,a)
    assert all(L.digest(v)==row['array_hashes'][key] for key,v in a.items())
    assert int(a['native_counts'].sum())==row['physics']
