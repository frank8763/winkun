"""Matched trajectory-coverage ablation, full-history task-balanced BC replay.

Uses the same T646 fullActor Adam/lr/steps, changes only successful-data coverage.
This is not canonical SIL/AWAC, and nominal replay is not a stability guarantee.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t646_offline_night'))
import t646_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
state,load,export,make,objective=Base.state,Base.load,Base.export,Base.make,Base.objective
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest


def schedule(old,new,k,p):
    tasks=sorted({g['id'] for g in new});assert len(tasks)==8
    selected=[]
    for task in tasks:
        pool=[g for g in new if g['id']==task]
        selected.append(pool[k%len(pool)])
    return [old[(k*p['old_batch']+i)%len(old)] for i in range(p['old_batch'])],selected


def fit(obj,old,new,p,guard,event):
    q,opt=obj;assert len(old)==384 and len(new)==(8 if p['arm']=='sparse' else 198)
    trace=[]
    for k in range(p['fit_steps']):
        guard();a,b=schedule(old,new,k,p);opt.zero_grad(set_to_none=True)
        loss,repair,keep=objective(q,a,b,dict(p,arm='full_command'));loss.backward()
        variables=[v for v in q.parameters() if v.requires_grad]
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k)
        assert all(torch.isfinite(v).all() for v in variables)
        trace.append(dict(step=k,old_ids=[g['id'] for g in a],new_sources=[g['source_sha256'] for g in b],
            loss=float(loss.detach()),repair=float(repair.detach()),keep=float(keep.detach()),gradient_norm=norm))
    return dict(Actor_Adam=len(trace),arm=p['arm'],new_trajectories=len(new),new_tasks=8,old_tasks=384,
        old_complete_passes=p['fit_steps']*p['old_batch']//384,full_causal_history=True,trace=trace)
