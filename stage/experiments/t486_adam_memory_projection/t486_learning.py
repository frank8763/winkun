"""Single targeted-memory pilot with projection of actual mature Adam steps.

Unchanged fresh PPO estimator. 15 replay histories, equal eight task-family
weights; eight individual old-left5 surrogate gradients constrain delta_theta.
Adam moments remain updated, as in projected optimization. Not a safety proof.
"""
from pathlib import Path
import re,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t481_fresh16_rehearsal'))
import t481_learning as F
from projection import project
B,P,E,M,R,torch,np=F.B,F.P,F.E,F.M,F.R,F.torch,F.np
read,save,sha,digest=F.read,F.save,F.sha,F.digest
make,batch,vtrace=F.make,F.batch,F.B.vtrace

def task(name):
    m=re.match(r'(?:dev|hold)_task([0-7])_',name);assert m,name
    return int(m[1])

def balanced(values,ids):
    assert len(values)==len(ids)
    groups=[[v for v,name in zip(values,ids) if task(name)==i] for i in range(8)]
    assert all(groups)
    return sum(sum(group)/len(group) for group in groups)/8

def references(p,own_rows,replay_rows):
    refs=B.H.references(p,own_rows);q=make(p)[0];device=next(q.parameters()).device;actual=[]
    with torch.no_grad():
        for r in replay_rows:
            assert sha(r['path'])==r['sha256'];a=E.N.S.arrays(r['path']);x=torch.tensor(a['states'],device=device)
            actual.append(dict(id=r['job_id'],x=x,target=q.sequence(x).detach().clone(),source_sha256=r['sha256']))
    assert len(actual)==len({g['id'] for g in actual}) and len(actual) in [8,15]
    refs['actual_old']=actual;refs['provenance']['actual_old']=[{k:g[k] for k in ['id','source_sha256']} for g in actual]
    refs['provenance']['replay_weight']='Eight taskfamilies equal; left5 eightepisodes share its one-eighth weight'
    return refs

def keep_loss(q,refs,k):
    _,old,own=B.H.keep_loss(q,refs,k);sd=next(q.parameters()).new_tensor(q.p['raw_sigma'])
    vals=[.5*((q.sequence(g['x'])-g['target'])/sd).square().mean() for g in refs['actual_old']]
    actual=balanced(vals,[g['id'] for g in refs['actual_old']])
    return .5*(.5*(old+actual)+own),old,actual,own

def memory(p,replay_rows,device):
    out=[]
    for r in replay_rows:
        assert sha(r['path'])==r['sha256'] and r['full']
        a=E.N.S.arrays(r['path']);assert not r['administrative_censor'] and a['stochastic'].all() and not a['ownership'].any()
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
        assert np.array_equal(a['states'][1:],a['next_states'][:-1])
        discount=p['gamma_native']**a['native_counts'].astype(np.int64);discount[-1]=0.
        out.append(dict(id=r['job_id'],source_sha256=r['sha256'],x=torch.tensor(a['states'],device=device),
            raw=torch.tensor(a['raw_actions'],device=device),atom=torch.tensor(a['atom'],device=device),behavior=torch.tensor(a['logp'],device=device),
            reward=torch.tensor(a['rewards'],device=device),discounts=torch.tensor(discount,device=device),locations=torch.tensor(a['locations'],device=device),terminal=[bool(x) for x in a['end_flags'][-1]]))
    assert len(out) in [8,15] and len(out)==len({b['id'] for b in out})
    assert {task(b['id']) for b in out}==set(range(8))
    return out

def replay_losses(q,v,data,p,assert_behavior=False,details=False):
    al=[];vl=[];stats=[];frozen_advantages=[]
    for b in data:
        loc=q.sequence(b['x']);lp=P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])
        if assert_behavior:
            assert float((loc-b['locations']).detach().abs().max())<1e-10
            assert float((lp-b['behavior']).detach().abs().max())<1e-9
        vv=v(b['x']);target,adv,w=vtrace(b['reward'],b['discounts'],torch.cat((vv,vv.new_zeros(1))),lp-b['behavior'])
        frozen_advantages.append(adv.detach().clone());al.append(-(lp*adv).mean());vl.append(.5*(vv-target).square().mean())
        stats.append(dict(id=b['id'],weight_mean=float(w.mean()),weight_min=float(w.min()),advantage_RMS=float(adv.square().mean().sqrt())))
    ids=[b['id'] for b in data];out=(balanced(al,ids),balanced(vl,ids),stats)
    return (*out,{name:dict(loss=loss,advantage=av,old_loss=float(loss.detach())) for name,loss,av in zip(ids,al,frozen_advantages)}) if details else out

def epoch(obj,items,refs,data,p,k,guard,event):
    guard();q,v,ao,vo=obj;ao.zero_grad(set_to_none=True);vo.zero_grad(set_to_none=True);pg=vl=0.
    for b in items:
        ratio=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp()
        loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/16;loss.backward();pg+=float(loss.detach())
        loss=.5*(v(b['x'])-b['target']).square().mean()/16;loss.backward();vl+=float(loss.detach())
    keep,old,actual,own=keep_loss(q,refs,k);(p['keep_weight']*keep).backward()
    al,cl,stats,by_id=replay_losses(q,v,data,p,details=True)
    # Differentiate each original empirical reference before Adam modifies q.
    ids=p.get('projection_fixture_ids',p['targeted_ids'])
    assert len(ids)==8 and len(set(ids))==8 and set(ids)<=set(by_id)
    params=list(q.parameters());g=[]
    for name in ids:
        grad=torch.autograd.grad(by_id[name]['loss'],params,retain_graph=True,allow_unused=False)
        g.append(torch.cat([z.detach().reshape(-1) for z in grad]).cpu().numpy())
    (p['replay_actor_weight']*al).backward();(p['replay_value_weight']*cl).backward()
    an=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));vn=float(torch.nn.utils.clip_grad_norm_(v.parameters(),1.,error_if_nonfinite=True))
    with torch.no_grad():before=torch.cat([z.detach().reshape(-1) for z in params]).clone()
    event('B','BEFORE',k);ao.step();vo.step()
    with torch.no_grad():
        proposed=torch.cat([z.detach().reshape(-1) for z in params])-before
        delta,projection=project(proposed.cpu().numpy(),np.stack(g))
        vector=before+before.new_tensor(delta);offset=0
        for z in params:z.copy_(vector[offset:offset+z.numel()].reshape_as(z));offset+=z.numel()
        applied=(torch.cat([z.detach().reshape(-1) for z in params])-before).cpu().numpy()
        norms=np.linalg.norm(g,axis=1);ix=norms>0
        actual_dots=(np.asarray(g)[ix]/norms[ix,None])@applied
        assert not len(actual_dots) or float(actual_dots.max())<=1e-10*max(float(np.linalg.norm(proposed.cpu().numpy())),1e-16)
        projection['actual_parameter_displacement_dots']=actual_dots.tolist();projection['reference_ids']=ids
        projection['nonlinear_frozen_surrogate_changes']={}
        for name in ids:
            record=next(b for b in data if b['id']==name)
            lp=P.L.density(q.sequence(record['x']),record['raw'],record['atom'],p['raw_sigma'])
            projection['nonlinear_frozen_surrogate_changes'][name]=float(-(lp*by_id[name]['advantage']).mean())-by_id[name]['old_loss']
        kl=sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/16
    event('B','AFTER',k)
    return dict(epoch=k,policy_loss=pg,value_loss=vl,keep=float(keep.detach()),old211_keep=float(old.detach()),
        actual_old15_keep=float(actual.detach()),own8_keep=float(own.detach()),replay_actor=float(al.detach()),replay_value=float(cl.detach()),
        replay_stats=stats,actor_norm=an,value_norm=vn,KL=kl,projection=projection,Adam_moments_retained_after_projection=True)

def fit(obj,arrays,p,refs,data,guard,event,steps=None):
    assert digest(P.state(*obj))==p['initial']['joint_hash']
    items,diagnostics=batch(obj,arrays,p);trace=[]
    for k in range(p['epochs'] if steps is None else steps):
        trace.append(epoch(obj,items,refs,data,p,k,guard,event))
        if trace[-1]['KL']>p['target_kl']:break
    return dict(trace=trace,epochs=len(trace),pooled_advantage=diagnostics,reference_projection_of_actual_Adam_displacement=True,
        single_pilot_NOT_causal_or_safety_proof=True,fresh_16_reused_NOT_new_independent=True)
