"""Matched data-support test: original-start versus own-state conditional PG.

Gaussian expected success is not a deterministic-control guarantee.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t654_verified_deviation_recovery'))
import t654_learning as C
sys.path.insert(0,str(ROOT/'stage/experiments/t651_canonical_outcome_pg'))
import t651_learning as PG
L,P,E,torch,np=C.L,C.P,C.E,C.torch,C.np
make,state,load,export=C.make,C.state,C.load,C.export
read,save,sha,digest=C.read,C.save,C.sha,C.digest
location,pack,keep_loss=C.location,C.pack,C.keep_loss


def features(q,a,row):
    device=next(q.parameters()).device;x=torch.tensor(a['states'],device=device)
    with torch.no_grad():
        n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);h=h[:,0]
        g=dict(id=row['job_id'],f=torch.cat([h,n],-1),logits=q.head(h),initial=q.sequence(x).detach(),source_sha256=row['sha256'])
        assert float((location(q,g)-g['initial']).abs().max())<1e-10
    return g


def build(q,groups,p):
    device=next(q.parameters()).device;items=[];diag=[]
    for key,group in groups:
        n=len(group);assert n in [4,8];rewards=[PG.outcome(row) for a,row in group];values=[]
        for i,(a,row) in enumerate(group):
            first=int(a.get('recovery_first',0));assert a['stochastic'][first:].all() and not a['ownership'][first:].any()
            cached=features(q,a,row)
            with torch.no_grad():
                loc=location(q,cached);assert float((loc-torch.tensor(a['locations'],device=device)).abs().max())<1e-10
                raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
                assert float((P.L.density(loc[first:],raw[first:],atom[first:],p['raw_sigma'])-old[first:]).abs().max())<1e-9
            baseline=sum(rewards[j] for j in range(n) if j!=i)/(n-1)
            values.append(dict(**cached,raw=raw,atom=atom,old=old,first=first,adv=rewards[i]-baseline))
        items.append(values);diag.append(dict(id=key,n=n,successes=sum(rewards),outcomes=rewards,advantages=[v['adv'] for v in values],conditional_suffix=any(v['first']>0 for v in values)))
    return items,diag


def task_loss(q,group,p):
    total=next(q.parameters()).new_zeros(())
    for g in group:
        k=g['first'];loc=location(q,g)[k:];ratio=(P.L.density(loc,g['raw'][k:],g['atom'][k:],p['raw_sigma'])-g['old'][k:]).exp()
        total-=torch.minimum(ratio*g['adv'],ratio.clamp(1-p['ratio_clip'],1+p['ratio_clip'])*g['adv']).sum()/(len(group)*p['score_scale'])
    return total


@torch.no_grad()
def trust(q,items,p):
    return [sum(float((.5*((location(q,g)[g['first']:]-g['initial'][g['first']:])/.2).square().sum(-1)).mean()) for g in group)/len(group) for group in items]


def fit(obj,old,new,p,guard,event):
    q,opt=obj;body=L.fixed_hash(q);device=next(q.parameters()).device
    items,diag=build(q,new['groups'],p);keep=pack(old,device);fields=pack(new['fields'],device);trace=[]
    assert len(old)==384 and len(new['fields'])==8 and len(items)==11
    for k in range(p['fit_steps']):
        guard();before=state(obj);opt.zero_grad(set_to_none=True);pg=0.
        for group in items:
            loss=task_loss(q,group,p)/len(items);loss.backward();pg+=float(loss.detach())
        a=keep_loss(q,keep);b=keep_loss(q,fields);(p['keep_weight']*a+p['new_keep_weight']*b).backward()
        variables=[v for v in q.parameters() if v.requires_grad];norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        event('BEFORE',k);opt.step();event('AFTER',k);kl=trust(q,items,p)
        okay=bool(np.isfinite(kl).all() and max(kl)<=p['max_task_latent_KL'])
        if not okay:load(obj,before)
        trace.append(dict(step=k,pg_loss=pg,old_keep=float(a.detach()),field_keep=float(b.detach()),norm=norm,per_task_KL=kl,accepted=okay))
        assert L.fixed_hash(q)==body and all(torch.isfinite(v).all() for v in variables)
        if not okay:assert digest(state(obj))==digest(before);break
    return dict(Actor_Adam=len(trace),accepted=sum(v['accepted'] for v in trace),arm=p['arm'],trace=trace,diagnostics=diag,
        trainable_parameters=11906,full384_each_update=True,all8_field_histories_each_update=True,encoder_GRU_head_frozen_exact=True,
        not_fresh_rollout_per_epoch=True,conditional_suffix_is_not_original_start_return=True,no_deterministic_success_guarantee=True)
