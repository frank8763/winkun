"""Joint supervised transfer of real successes, with native protection checks.

This is off-policy success consolidation, not another on-policy PPO update.
"""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t504_success_replay'));import t504_learning as B
A,G,P,torch,np=B.A,B.G,B.P,B.torch,B.np
read,sha,digest=G.read,G.sha,G.digest
make=B.make

def groups(p):
    q=make(p)[0].requires_grad_(False);data=[]
    for kind,key in [('repair','repair_rows'),('keep','keep_rows')]:
        for index,row in enumerate(p[key]):
            assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256']
            a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
            assert not a['ownership'].any()
            assert a['stochastic'].all() if kind=='repair' else not a['stochastic'].any()
            g=B.cached(q,a)
            assert float((g['target']-torch.tensor(a['locations'])).abs().max())<1e-10
            if kind=='repair':
                with torch.no_grad():
                    err=float((P.L.density(g['target'],torch.tensor(a['raw_actions']),torch.tensor(a['atom']),p['raw_sigma'])-torch.tensor(a['logp'])).abs().max());assert err<1e-9
                target=torch.tensor(a['actions'],dtype=torch.float64)
                goal=a['task_goal_xy'].reshape(-1,2)[0];near=np.linalg.norm(a['poses'][:,:2]-goal,axis=1)<=p['near_goal_radius_m'];assert near.any()
            else:target=torch.tensor(a['locations'])
            family=p['new_task_ids'].index(row['job_id']) if kind=='repair' else int(row['job_id'].startswith('t502_'))
            g.update(kind=kind,family=family,id=row['job_id'],record_id=kind+'_'+str(index),target=target,source_sha256=row['sha256'],dt=torch.tensor(a['native_counts'].astype(float)))
            if kind=='repair':g['near']=torch.tensor(near)
            data.append(g)
    assert all(sum(g['kind']=='repair' and g['family']==i for g in data)>=1 for i in range(2))
    assert [sum(g['kind']=='keep' and g['family']==i for g in data) for i in range(2)]==[8,11]
    return data

def losses(q,g,p):
    loc=A.location(q,g);sigma=loc.new_tensor(p['raw_sigma'])
    axis=.5*P.action_loss(loc,g['target'])/sigma.square() if g['kind']=='repair' else .5*((loc-g['target'])/sigma).square()
    err=axis.mean(-1)
    if g['kind']=='repair':
        near=g['near'];dt=g['dt']
        return .5*(err*dt).sum()/dt.sum()+.5*(err[near]*dt[near]).sum()/dt[near].sum()
    return .5*(err*g['dt']).sum()/g['dt'].sum()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean()

def measure(q,data,p):
    with torch.no_grad():return {g['record_id']:float(losses(q,g,p)) for g in data}

def fit(obj,data,p,guard,event,steps=None):
    q,v,ao,vo=obj;fixed=A.fixed_hash(q);vh=digest(v.state_dict());voh=digest(vo.state_dict());before=[int(ao.state[z]['step']) for z in A.variables(q)]
    initial=measure(q,data,p);trace=[]
    for k in range(p['block_steps'] if steps is None else steps):
        guard();ao.zero_grad(set_to_none=True);total=dict(repair=0.,keep=0.)
        for g in data:
            families=2;count=sum(h['kind']==g['kind'] and h['family']==g['family'] for h in data)
            loss=losses(q,g,p)/(families*count);(.5*loss).backward();total[g['kind']]+=float(loss.detach())
        norm=float(torch.nn.utils.clip_grad_norm_(A.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);ao.step();event('AFTER',k)
        trace.append(dict(step=k,successful_action_loss=total['repair'],actual15_history_keep=total['keep'],gradient_norm=norm))
        assert A.fixed_hash(q)==fixed and digest(v.state_dict())==vh and digest(vo.state_dict())==voh
    assert [int(ao.state[z]['step']) for z in A.variables(q)]==[n+len(trace) for n in before]
    return dict(Actor_Adam=len(trace),Value_Adam=0,new_policy_gradient_updates=0,initial=initial,final=measure(q,data,p),trace=trace,trainable_parameters=11906)
