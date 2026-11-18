"""Task-balanced verified recovery aggregation; original full384 mean keep."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t637_joint_coverage_recovery'))
import t637_learning as Base
L,P,E,torch,np=Base.L,Base.P,Base.E,Base.torch,Base.np
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest
make,state,load,pack,correction,location,export=Base.make,Base.state,Base.load,Base.pack,Base.correction,Base.location,Base.export


def objective(q,keep,corrections,p):
    retention=((location(q,keep)-keep['target'])/.2).square().mean(-1)
    protection=(retention*keep['weights']).sum()
    counts={g['id']:sum(x['id']==g['id'] for x in corrections) for g in corrections}
    repair=protection.new_zeros(())
    for g in corrections:
        err=P.action_loss(location(q,g),g['target']).mean(-1)/.2**2
        value=.5*err.mean()+.5*err.topk(max(1,(len(err)+4)//5)).values.mean()
        repair+=value/(len(counts)*counts[g['id']])
    return repair+p['keep_weight']*protection,repair,protection


def fit(obj,keep,corrections,p,guard,event):
    previous=Base.objective;Base.objective=objective
    try:
        # Original fitter's end metrics expect one correction dictionary. The
        # optimizer loop is reproduced explicitly to keep per-task diagnostics.
        q,opt=obj;fixed=L.fixed_hash(q);trace=[]
        for k in range(p['fit_steps']):
            guard();opt.zero_grad(set_to_none=True);loss,r,s=objective(q,keep,corrections,p);loss.backward()
            norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True))
            event('BEFORE',k);opt.step();event('AFTER',k)
            assert all(torch.isfinite(v).all() for v in L.variables(q))
            trace.append(dict(step=k,loss=float(loss.detach()),repair=float(r.detach()),keep=float(s.detach()),norm=norm))
        assert fixed==L.fixed_hash(q)
        with torch.no_grad():
            err=(location(q,keep)-keep['target']).abs()
            metrics=dict(max_keep_raw_error=float(err.max()),
                max_keep_command_error=float((location(q,keep).clamp(-1,1)-keep['target'].clamp(-1,1)).abs().max()),
                corrections=[dict(id=g['id'],source_sha256=g['source_sha256'],
                   RMSE=float((location(q,g).clamp(-1,1)-g['target']).square().mean().sqrt())) for g in corrections])
        return dict(Actor_Adam=len(trace),Value_Adam=0,full384_history_equal_weight=True,
            recovery_tasks_equal_weight=True,trainable_parameters=11906,trace=trace,metrics=metrics)
    finally:Base.objective=previous
