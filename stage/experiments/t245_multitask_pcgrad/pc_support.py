"""Fresh four-critical-reset MC gradients with deterministic PCGrad surgery."""
from pathlib import Path
import sys,importlib.util
_ROOT=Path(__file__).resolve().parents[3];_spec=importlib.util.spec_from_file_location('stage239_pcbase',_ROOT/'stage/experiments/t239_ar_policy_gradient/ar_support.py');_A=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_A;_spec.loader.exec_module(_A)
from stage239_pcbase import *
ROOT=_ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t245_multitask_pcgrad'
def worker(pipe,index,deadline):_A.OUT=OUT;_A.worker(pipe,index,deadline)
def anchors(plan,p):
    with np.load(plan['induced_states']) as z:xx=z['states'].copy();off=z['offsets'];ok=z['success'];task=z['task']
    groups=[torch.tensor(np.concatenate([xx[off[k]:off[k+1]] for k in range(211) if ok[k] and task[k]==t])[::plan['anchor_stride']]) for t in range(8)]
    x=torch.cat(groups);offset=np.cumsum([0]+[len(g) for g in groups]).tolist()
    with torch.no_grad():y=p(x).detach()
    return x,y,offset
def batch(items,p,plan):
    d=_A.batch(items,p,plan);names=sorted({i['row']['source_reset_id'] for i in items});d['task_labels']=torch.tensor(np.concatenate([np.full(len(i['arrays']['states']),names.index(i['row']['source_reset_id']),dtype=np.int64) for i in items]));d['task_count']=len(names);return d
def project(gradients,seed):
    rng=np.random.default_rng(seed);result=[];conflicts=0
    for i,gi in enumerate(gradients):
        q=gi.clone()
        for j in rng.permutation(len(gradients)):
            if i==j:continue
            g=gradients[j];dot=torch.dot(q,g)
            if float(dot)<0:q-=dot/g.square().sum().clamp_min(1e-30)*g;conflicts+=1
        result.append(q)
    return torch.stack(result).mean(0),conflicts
def update(p,opt,data,ax,ay,offset,plan,round_index,guard):
    rng=np.random.default_rng(plan['seed']+round_index);ind=np.concatenate([rng.integers(offset[i],offset[i+1],plan['anchor_batch_each']) for i in range(8)]);sx=ax[ind];sy=ay[ind];params=[q for q in p.parameters() if q.requires_grad];reports=[]
    flat=lambda gg:torch.cat([g.reshape(-1) for g in gg])
    def measure():
        m=p.physical_mean(data['x']);lp=AR.logp(m,data['actions'],m[data['prev']],data['actions'][data['prev']],data['reset']);delta=lp-data['oldlp'];ratio=delta.exp();a=data['advantage'];terms=torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a)*data['weights'];surrogates=[terms[data['task_labels']==t].sum()/4/234 for t in range(data['task_count'])];surr=torch.stack(surrogates).mean();ret=plan['retention_weight']*(p(sx)-sy).square().mean();return surr-ret,surr,delta,surrogates,ret
    for k in range(plan['actor_attempts_per_round']):
        guard();obj,_,_,surrogates,ret=measure();before=float(obj.detach());grads=[flat(torch.autograd.grad(-s,params,retain_graph=True)) for s in surrogates];retg=flat(torch.autograd.grad(ret,params));combined,conflicts=project(grads,plan['seed']+100*round_index+k);combined+=retg;opt.zero_grad();pos=0
        for q in params:q.grad=combined[pos:pos+q.numel()].view_as(q).clone();pos+=q.numel()
        first=float(p.body.net[0].weight.grad.norm());gn=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
        def evaluate(q):
            try:o,s,d,_,_=measure()
            except AssertionError as e:return dict(accept=False,numeric_domain_rejection=str(e))
            err=(q(ax)-ay).abs();mse=torch.stack([err[offset[i]:offset[i+1]].square().mean() for i in range(8)]).mean();q99=torch.quantile(err.flatten(),.99);mx=err.max();k3=(d.exp()-1-d).mean();dl=d.abs().max();ok=bool(torch.isfinite(o)) and float(o)>before+1e-10 and float(k3)<=plan['max_conditional_k3'] and float(dl)<=plan['max_abs_log_ratio'] and float(mse)<=plan['max_anchor_MSE'] and float(q99)<=plan['max_anchor_q99'] and float(mx)<=plan['max_anchor_error']
            return dict(accept=ok,objective=float(o),surrogate=float(s),improvement=float(o)-before,conditional_empirical_k3=float(k3),max_abs_log_ratio=float(dl),balanced_anchor_MSE=float(mse),anchor_q99=float(q99),anchor_max=float(mx))
        result=guarded_adam_step(p,opt,evaluate,plan['step_factors']);result.update(step=k,pcgrad_negative_projection_count=conflicts,first_layer_gradient_norm=first,gradient_norm_before_clip=gn,policy_hash=digest(p.state_dict()),optimizer_hash=digest(opt.state_dict()));reports.append(result)
        if not result['accepted']:break
    return reports
