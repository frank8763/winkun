"""Matched neural state-V/MC versus SMDP-GAE pilot; no Q action gradients.

Cross-fitting is by complete innovation trajectories, never time-row leakage.
The finite sample variance checks are engineering screens, not confidence bounds.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'stage/experiments/t309_sequence_goal_value'))
from sequence_value import *
sys.path.insert(0,str(ROOT/'stage/experiments/goal_conditioned_ppo'))
from advantage import advantage
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t312_online_goal_ppo';DOC=ROOT/'stage/docs/distance_20261103'
VALUE_INFLIGHT={}

class Value(torch.nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(seed)
        self.register_buffer('mu',torch.zeros(120));self.register_buffer('sd',torch.ones(120))
        self.net=torch.nn.Sequential(torch.nn.Linear(120,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,1))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,x):return self.net(((x-self.mu)/self.sd).clamp(-10,10)).squeeze(-1)

def policy(p):
    q=B.actor(p);q.requires_grad_(True);q.base.logstd.requires_grad_(False);return q

def means(q,x):return torch.cat([q.step(row[None])[1] for row in x])
def lp(q,a,index=None):
    x=torch.tensor(a['states'] if index is None else a['states'][index]);ac=torch.tensor(a['actions'] if index is None else a['actions'][index])
    return D.terms(means(q,x),ac)[1].sum(-1)

def fit_values(p,items,rnd,guard,event):
    VALUE_INFLIGHT.clear()
    values=[None]*len(items);models=[];optimizers=[];records=[]
    assert len(items)==16 and len({i['row']['goal_family'] for i in items})==8
    for fold in [0,1]:
        train=[i for i in items if i['row']['draw']!=fold];test=[(k,i) for k,i in enumerate(items) if i['row']['draw']==fold]
        assert len(train)==len(test)==8
        seed=p['seed']+100*rnd+fold;model=Value(seed);opt=torch.optim.Adam(model.parameters(),lr=p['value_lr'])
        VALUE_INFLIGHT[fold]=(model,opt)
        x=np.concatenate([i['arrays']['states'] for i in train]);w=np.concatenate([np.full(len(i['arrays']['states']),1/(8*len(i['arrays']['states']))) for i in train])
        mu=(x*w[:,None]).sum(0);sd=np.sqrt(np.maximum(((x-mu)**2*w[:,None]).sum(0),1e-6))
        model.mu.copy_(torch.tensor(mu,dtype=torch.float32));model.sd.copy_(torch.tensor(sd,dtype=torch.float32));rng=np.random.default_rng(seed)
        for step in range(p['value_steps']):
            guard();indices=[rng.integers(0,len(i['returns']),16) for i in train]
            xx=torch.tensor(np.concatenate([i['arrays']['states'][ix] for i,ix in zip(train,indices)]));yy=torch.tensor(np.concatenate([i['returns'][ix] for i,ix in zip(train,indices)]),dtype=torch.float32)
            opt.zero_grad();loss=(model(xx)-yy).square().mean();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            event('V',dict(round=rnd,fold=fold,step=step));opt.step()
        with torch.no_grad():
            for k,i in test:
                a=i['arrays'];xx=np.concatenate([a['states'],a['next_states'][-1:]])
                values[k]=model(torch.tensor(xx)).double().numpy()
        models.append(model);optimizers.append(opt);records.append(dict(fold=fold,train_ids=[i['row']['id'] for i in train],test_ids=[i['row']['id'] for k,i in test],model_hash=S.digest(model.state_dict()),optimizer_hash=S.digest(opt.state_dict()),steps=p['value_steps'],last_train_loss=float(loss.detach())))
    return values,models,optimizers,records

def build(p,items,values):
    target=[];group={};scale=float(np.sqrt(np.mean([np.mean(i['returns']**2) for i in items])));scale=max(.05,scale)
    for i,v in zip(items,values):
        a=i['arrays'];mc=i['returns']-v[:-1]
        gae=advantage(a['rewards'],a['native_counts'],v,a['end_flags'],gamma=R.GAMMA,lam=p['lambda'])['advantage']
        assert np.isfinite(mc).all() and np.isfinite(gae).all()
        target.append(dict(mc=mc,gae=gae,zero=i['returns'],prefix=R.GAMMA**np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))[:-1]]))
        g=group.setdefault(i['row']['goal_family'],dict(zero=[],state=[]))
        g['zero'].append(float(np.mean(i['returns']**2)));g['state'].append(float(np.mean(mc**2)))
    metrics={k:dict(zero_MSE=float(np.mean(v['zero'])),state_MSE=float(np.mean(v['state']))) for k,v in group.items()}
    zero=float(np.mean([v['zero_MSE'] for v in metrics.values()]));state=float(np.mean([v['state_MSE'] for v in metrics.values()]))
    gate=state<=.9*zero and all(v['state_MSE']<=1.1*v['zero_MSE']+1e-12 for v in metrics.values())
    return target,dict(equal_goal_zero_MSE=zero,equal_goal_state_MSE=state,per_goal=metrics,common_zero_MC_RMS_scale=scale,heldout_value_gate=bool(gate))

def gradient_screen(p,q,items,target,guard):
    params=[v for v in q.parameters() if v.requires_grad];vectors=[];hashes=[];density=[]
    before=S.digest(q.state_dict())
    for i,t in zip(items,target):
        guard();a=i['arrays'];logp=lp(q,a)
        with torch.no_grad():m=means(q,torch.tensor(a['states']))
        err=float((logp-torch.tensor(a['logp'])).abs().max().detach());me=float((m-torch.tensor(a['means'])).abs().max())
        assert err==0 and me==0,(err,me)
        for name in ['zero','mc','gae']:
            coefficient=torch.tensor(t['prefix']*t[name]/280,dtype=torch.float64)
            g=torch.autograd.grad(torch.dot(logp,coefficient),params,retain_graph=name!='gae',allow_unused=True)
            v=torch.cat([torch.zeros_like(param).reshape(-1) if grad is None else grad.detach().reshape(-1) for param,grad in zip(params,g)])
            assert torch.isfinite(v).all();vectors.append(v);hashes.append(dict(id=i['row']['id'],method=name,sha256=S.digest(v)))
        density.append(dict(id=i['row']['id'],mean_error=me,logp_error=err))
    matrix=torch.stack(vectors).double();gram=(matrix@matrix.T).numpy();per={}
    for family in sorted({i['row']['goal_family'] for i in items}):
        ids=[k for k,i in enumerate(items) if i['row']['goal_family']==family];assert len(ids)==2
        r={}
        for axis,name in enumerate(['zero','mc','gae']):
            ix=[3*k+axis for k in ids];g=gram[np.ix_(ix,ix)];r[name]=max(0.,float(np.trace(g)-g.sum()/2))
        per[family]=r
    # A factor-two pilot guard explicitly rejects local explosions like T292.
    variance_pass=all(v[name]<=2*v['zero']+1e-12 for v in per.values() for name in ['mc','gae'])
    u=np.zeros(48);v=u.copy();u[1::3]=1/16;v[2::3]=1/16
    denom=np.sqrt(max(0,float(u@gram@u))*max(0,float(v@gram@v)))
    cosine=float(u@gram@v/denom) if denom>1e-30 else None
    assert S.digest(q.state_dict())==before and all(v.grad is None for v in params)
    return dict(per_goal_gradient_sample_variance=per,finite_sample_GAE_MC_mean_cosine=cosine,variance_gate=bool(variance_pass),gradient_hashes=hashes,density=density),gram

def actor_steps(p,q,opt,items,target,arm,rnd,ax,ay,offset,guard,event):
    report=[];params=[v for v in q.parameters() if v.requires_grad];scale=max(.05,float(np.sqrt(np.mean([np.mean(i['returns']**2) for i in items]))))
    for step in range(p['actor_steps']):
        guard();rng=np.random.default_rng(p['seed']+1000*rnd+step);idx=[rng.integers(0,len(i['returns']),16) for i in items]
        old=np.concatenate([i['arrays']['logp'][ix] for i,ix in zip(items,idx)]);adv=np.concatenate([t[arm][ix]*t['prefix'][ix]/scale for t,ix in zip(target,idx)])
        new=torch.cat([lp(q,i['arrays'],ix) for i,ix in zip(items,idx)]);delta=new-torch.tensor(old);ratio=delta.exp();a=torch.tensor(adv)
        surrogate=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
        oi=np.concatenate([rng.integers(offset[k],offset[k+1],16) for k in range(8)])
        keep=(q.step(ax[oi])[0]-ay[oi]).square().mean()*p['retention_weight'];loss=-surrogate+keep
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        event('Actor',dict(round=rnd,arm=arm,step=step));opt.step();opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            ds=torch.cat([lp(q,i['arrays'])-torch.tensor(i['arrays']['logp']) for i in items]);k3=float((ds.exp()-1-ds).mean());maxdelta=float(ds.abs().max())
        numeric=np.isfinite(k3) and np.isfinite(maxdelta) and k3<=p['max_conditional_k3'] and maxdelta<=p['max_abs_log_ratio']
        row=dict(step=step,loss=float(loss.detach()),surrogate=float(surrogate.detach()),weighted_retention=float(keep.detach()),gradient_norm=norm,k3=k3,max_abs_logratio=maxdelta,numeric_pass=bool(numeric),policy_hash=S.digest(q.state_dict()),optimizer_hash=S.digest(opt.state_dict()))
        report.append(row)
        if not numeric:break
    return report

def compact(a):return {k:a[k] for k in ['actions','native_counts','end_flags','environment_rewards']}

def load_items(rows):
    return [dict(row=r,arrays=B.arrays(r['path']),returns=S.monte_carlo(B.arrays(r['path'])['rewards'],B.arrays(r['path'])['native_counts'])) for r in rows]
