"""Full-episode causal recurrent policy gradients with matched memory-off control."""
from mem_runtime import *
import copy
sys.path.insert(0,str(ROOT/'stage/experiments'))
from native_update_guard import guarded_adam_step

def retention(plan,parent):
    with np.load(plan['induced_states']) as z:
        xx=z['states'].copy();off=z['offsets'];ok=z['success'];task=z['task'];ids=z['case_ids']
    with np.load(plan['steering_context']) as z:
        assert np.array_equal(off,z['offsets']) and np.array_equal(ids,z['case_ids']);enc=z['encoders'].copy()
    rows={r['job']['source_reset_id']:r for r in map(json.loads,Path(plan['retention_rows']).read_text().splitlines())}
    selected=[];items=[]
    for t in range(8):
        kk=np.flatnonzero(ok & (task==t));chosen=[int(kk[0]),int(kk[len(kk)//2]),int(kk[-1])]
        for k in chosen:
            name=str(ids[k]);r=rows[name]
            with np.load(r['path']) as z:
                act=z['actions'].copy();counts=z['native_counts'].copy()
                assert digest(act)==r['array_hashes']['actions'] and digest(counts)==r['array_hashes']['native_counts']
            x117=np.concatenate([xx[off[k]:off[k+1]],enc[off[k]:off[k+1]]],axis=1);n=len(x117);assert len(act)==n
            prev=np.concatenate([np.zeros((1,2),dtype=np.float32),act[:-1]]);dt=np.r_[0,counts[:-1]].astype(np.float32)[:,None]/100
            x=np.concatenate([x117,prev,dt],axis=1).astype(np.float32);items.append(dict(states=x,actions=act));selected.append(name)
    d=pad(items)
    with torch.no_grad():target=parent(d['x'][...,:117].reshape(-1,117).contiguous()).reshape(*d['mask'].shape,2)
    d['target']=target;d['ids']=selected
    return d

def retention_error(p,anchor):
    mode,_,_=p.sequence(anchor['x']);e=(mode-anchor['target']).abs();m=anchor['mask']
    # Three complete causal episodes per task; task weights remain equal.
    ms=torch.stack([e[:,i:i+3][m[:,i:i+3]].square().mean() for i in range(0,24,3)]).mean()
    return ms,e[m].max()

def project(gradients,seed):
    rng=np.random.default_rng(seed);out=[];conflicts=0
    for i,gi in enumerate(gradients):
        g=gi.clone()
        for j in rng.permutation(len(gradients)):
            if i==j:continue
            dot=torch.dot(g,gradients[j])
            if float(dot)<0:g-=dot/gradients[j].square().sum().clamp_min(1e-30)*gradients[j];conflicts+=1
        out.append(g)
    return torch.stack(out).mean(0),conflicts

def optimizer(p,plan):
    return torch.optim.Adam([dict(params=[q for q in p.base.parameters() if q.requires_grad],lr=plan['base_lr']),dict(params=list(p.encoder.parameters())+list(p.core.parameters())+list(p.residual.parameters()),lr=plan['memory_lr'])])

def update(p,opt,data,anchor,plan,seed,rnd,guard):
    params=[q for q in p.parameters() if q.requires_grad];reports=[]
    flat=lambda gg:torch.cat([g.reshape(-1) for g in gg])
    def measure():
        lp,_=likelihood(p,data);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];term=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights']
        ss=[term[data['task']==t].sum()/4/234 for t in range(4)];surr=torch.stack(ss).mean();mse,_=retention_error(p,anchor);ret=plan['retention_weight']*mse
        return surr-ret,surr,delta,ss,ret
    for k in range(plan['actor_attempts_per_round']):
        guard();obj,_,_,surrogates,ret=measure();before=float(obj.detach())
        gs=[flat(torch.autograd.grad(-s,params,retain_graph=True)) for s in surrogates];rg=flat(torch.autograd.grad(ret,params));g,conflicts=project(gs,seed+100*rnd+k);g+=rg;opt.zero_grad();pos=0
        for q in params:q.grad=g[pos:pos+q.numel()].reshape_as(q).clone();pos+=q.numel()
        gradients={name:float(q.grad.norm()) for name,q in p.named_parameters() if q.requires_grad and name in ['core.weight_ih_l0','core.weight_hh_l0','encoder.0.weight','residual.weight','base.body.net.0.proprio.weight']}
        gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        def evaluate(q):
            try:o,s,d,_,_=measure()
            except AssertionError as e:return dict(accept=False,numeric_domain_rejection=str(e))
            mse,mx=retention_error(q,anchor);k3=(d.exp()-1-d).mean();dl=d.abs().max()
            good=bool(torch.isfinite(o)) and float(o)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio'] and float(mse)<=plan['max_anchor_MSE'] and float(mx)<=plan['max_anchor_error']
            return dict(accept=good,objective=float(o),surrogate=float(s),improvement=float(o)-before,conditional_k3=float(k3),max_abs_log_ratio=float(dl),anchor_MSE=float(mse),anchor_max=float(mx))
        r=guarded_adam_step(p,opt,evaluate,plan['step_factors']);r.update(step=k,gradient_norm=gn,gradient_groups=gradients,pcgrad_conflicts=conflicts,policy_hash=digest(p.state_dict()),optimizer_hash=digest(opt.state_dict()));reports.append(r)
        if not r['accepted']:break
    return reports
