"""New-distance actual policy gradients; preserve the accepted 5m function."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t255_overnight'))
import learning as L
M=L;Actor,C,R,S,np,torch=L.Actor,L.C,L.R,L.S,L.np,L.torch
def policy(plan):
    p=Actor(L.read(plan['initial_base'])['policy'],False,2026103501);p.load_state_dict(L.read(plan['checkpoint'])['policy']);return p
def anchors(plan,parent):
    with np.load(plan['retention_states']) as z:
        xx=z['states'];off=z['offsets'];task=z['task'];ids=z['case_ids'];groups=[torch.tensor(np.concatenate([xx[off[k]:off[k+1]] for k in range(211) if task[k]==t])) for t in range(8)]
    x=torch.cat(groups);offset=np.cumsum([0]+[len(g) for g in groups]).tolist();targets=[]
    with torch.no_grad():
        for chunk in x.split(2048):targets.append(parent.step(chunk)[0])
    return x,torch.cat(targets),offset
def anchor_measure(p,x,y,offset):
    errors=[]
    for chunk,target in zip(x.split(2048),y.split(2048)):errors.append((p.step(chunk)[0]-target).abs())
    e=torch.cat(errors);return torch.stack([e[offset[i]:offset[i+1]].square().mean() for i in range(8)]).mean(),e.max()
def update(p,opt,data,x,y,offset,plan,rnd,guard):
    rng=np.random.default_rng(plan['seed']+rnd);indices=np.concatenate([rng.integers(offset[t],offset[t+1],32) for t in range(8)]);sx=x[indices];sy=y[indices];params=[q for q in p.parameters() if q.requires_grad];reports=[];flat=lambda gs:torch.cat([g.reshape(-1) for g in gs])
    def measure():
        lp,_=L.likelihood(p,data);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];term=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights'];ss=[term[data['task']==t].sum()/4/plan['surrogate_horizon_decisions'] for t in range(4)];s=torch.stack(ss).mean();ret=plan['retention_weight']*(p.step(sx)[0]-sy).square().mean();return s-ret,s,delta,ss,ret
    for k in range(plan['actor_attempts_per_round']):
        guard();o,_,_,ss,ret=measure();before=float(o.detach());gs=[flat(torch.autograd.grad(-s,params,retain_graph=True)) for s in ss];rg=flat(torch.autograd.grad(ret,params));g,conflicts=L.project(gs,plan['seed']+100*rnd+k);g+=rg;opt.zero_grad();pos=0
        for q in params:q.grad=g[pos:pos+q.numel()].reshape_as(q).clone();pos+=q.numel()
        gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        def evaluate(q):
            try:o,s,d,_,_=measure()
            except AssertionError as e:return dict(accept=False,numeric_domain_rejection=str(e))
            mse,mx=anchor_measure(q,x,y,offset);k3=(d.exp()-1-d).mean();dl=d.abs().max();good=bool(torch.isfinite(o)) and float(o)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio'] and float(mse)<=plan['max_anchor_MSE'] and float(mx)<=plan['max_anchor_error']
            return dict(accept=good,objective=float(o),surrogate=float(s),improvement=float(o)-before,conditional_k3=float(k3),max_abs_log_ratio=float(dl),balanced_anchor_MSE=float(mse),anchor_max=float(mx))
        r=L.guarded_adam_step(p,opt,evaluate,plan['step_factors']);r.update(step=k,gradient_norm=gn,pcgrad_conflicts=conflicts,policy_hash=S.digest(p.state_dict()),optimizer_hash=S.digest(opt.state_dict()));reports.append(r)
        if not r['accepted']:break
    return reports
