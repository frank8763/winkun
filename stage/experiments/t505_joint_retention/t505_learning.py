"""Joint old-command repair with empirical per-ID native retention guards.

No planner/teacher at deployment. Frozen recurrent features use complete carry.
"""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t504_success_replay'))
import t504_learning as B
A,G,P,torch,np=B.A,B.G,B.P,B.torch,B.np
read,sha,digest=G.read,G.sha,G.digest
make=B.make

def groups(p):
    q=make(p)[0].requires_grad_(False);data=[]
    for kind,key in [('repair','repair_rows'),('keep','keep_rows')]:
        for row in p[key]:
            assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
            a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
            assert not a['ownership'].any() and not a['stochastic'].any()
            g=B.cached(q,a);target=torch.tensor(a['locations'])
            if kind=='keep':assert float((g['target']-target).abs().max())<1e-10
            g.update(kind=kind,id=row['job_id'],target=target,source_sha256=row['sha256'],dt=torch.tensor(a['native_counts'].astype(float)))
            data.append(g)
    assert sum(g['kind']=='repair' for g in data)==2 and sum(g['kind']=='keep' for g in data)==13
    assert len({g['id'] for g in data})==15
    return data

def losses(q,g,p):
    loc=A.location(q,g);err=.5*((loc-g['target'])/loc.new_tensor(p['raw_sigma'])).square().mean(-1)
    return .5*(err*g['dt']).sum()/g['dt'].sum()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean()

def measure(q,data,p):
    with torch.no_grad():return {g['id']:float(losses(q,g,p)) for g in data}

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=A.fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());before=[int(ao.state[z]['step']) for z in A.variables(q)]
    counts={k:sum(g['kind']==k for g in data) for k in ['repair','keep']};initial=measure(q,data,p);trace=[]
    for k in range(p['block_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);totals=dict(repair=0.,keep=0.)
        for g in data:
            loss=losses(q,g,p)/counts[g['kind']];(.5*loss).backward();totals[g['kind']]+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(A.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        trace.append(dict(step=k,repair_loss=totals['repair'],keep_loss=totals['keep'],gradient_norm=norm))
        assert A.fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    assert [int(ao.state[z]['step']) for z in A.variables(q)]==[n+len(trace) for n in before]
    return dict(Actor_Adam=len(trace),Value_Adam=0,new_policy_gradient_updates=0,initial=initial,final=measure(q,data,p),trace=trace,trainable_parameters=11906)
