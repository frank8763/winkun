"""One finite joint-outcome candidate; sampled constraints are not guarantees."""
from pathlib import Path
import copy,json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t670_outcome_gradient_gate'));import t670_analysis as A
M,torch,np=A.M,A.torch,A.np
read,sha,digest=M.read,M.sha,M.digest


def data(p,q,guard):
    oldrows=[r for r in read(p['old_summary'])['rows'] if not r.get('exact')]
    old=[]
    for key in p['old_ids']:
        guard();g,y=A.group(q,[r for r in oldrows if r['job_id']==key],p);assert len(g)==8;old.append(g)
    training=read(p['fresh_rounds'])[0]['training_rows'];field=[]
    for job in p['field_jobs']:
        guard();g,y=A.group(q,[r for r in training if r['job_id']==job['id']],p);assert len(g)==2;field.append(g)
    for batch in old+field:
        for g in batch:
            with torch.no_grad():g['initial']=M.location(q,g).detach().clone()
    refs=M.references(p,q,guard);return old,field,M.pack(refs)


def pg(q,g,p):
    loss=next(q.parameters()).new_zeros(())
    for item in g:
        ratio=(M.P.L.density(M.location(q,item),item['raw'],item['atom'],p['raw_sigma'])-item['old']).exp()
        adv=item['adv']
        loss-=(torch.minimum(ratio*adv,ratio.clamp(1-p['ratio_clip'],1+p['ratio_clip'])*adv)-adv).sum()/(len(g)*p['score_scale'])
    return loss


def optimizer_state(q,opt):return copy.deepcopy(dict(actor=q.state_dict(),actor_opt=opt.state_dict()))


def load(q,opt,state):
    q.load_state_dict(copy.deepcopy(state['actor']));opt.load_state_dict(copy.deepcopy(state['actor_opt']))


def fit(p,q,opt,old,field,keep,guard,event):
    variables=[v for v in q.parameters() if v.requires_grad];fixed=M.L.fixed_hash(q);trace=[]
    initial_old=[float(A.outcome_surrogate(q,g,p).detach()) for g in old]
    for k in range(p['Actor_steps']):
        guard();before=optimizer_state(q,opt);old_before=[float(A.outcome_surrogate(q,g,p).detach()) for g in old]
        opt.zero_grad(set_to_none=True);goal=0.
        for g in field+old:
            term=pg(q,g,p)/len(field+old);term.backward();goal+=float(term.detach())
        keep_loss,losses=M.retention(q,keep,False);(p['replay_weight']*keep_loss).backward()
        norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
        gradient=torch.cat([v.grad.detach().flatten() for v in variables]);theta_before=torch.cat([v.detach().flatten().clone() for v in variables])
        event('BEFORE',k);opt.step();event('AFTER',k)
        theta_after=torch.cat([v.detach().flatten() for v in variables]);delta=theta_after-theta_before
        formula_error=None
        if k==0:
            expected=-p['fit_lr']*gradient/(gradient.abs()+p['Adam_eps'])
            formula_error=float((delta-expected).abs().max());assert formula_error<1e-12
        old_after=[float(A.outcome_surrogate(q,g,p).detach()) for g in old]
        sampled_old_ok=all(a>=b-p['old_surrogate_tolerance'] for a,b in zip(old_after,old_before))
        finite=all(bool(torch.isfinite(v).all()) for v in variables)
        kl=[]
        with torch.no_grad():
            for group in field+old:
                kl.append(sum(float((.5*((M.location(q,g)-g['initial'])/.2).square().sum(-1)).mean()) if 'initial' in g else 0. for g in group)/len(group))
        accepted=sampled_old_ok and finite and max(kl)<=p['max_task_latent_KL']
        if not accepted:load(q,opt,before);assert digest(optimizer_state(q,opt))==digest(before)
        assert M.L.fixed_hash(q)==fixed
        trace.append(dict(step=k,pg=goal,keep=float(keep_loss.detach()),worst_reference_loss=float(losses.max().detach()),norm=norm,
            old_before=old_before,old_after=old_after,sampled_old_ok=sampled_old_ok,kl=kl,accepted=accepted,
            actual_displacement_sha256=digest(delta),actual_displacement_norm=float(delta.norm()),first_Adam_formula_error=formula_error))
        if not accepted:break
    return dict(Actor_Adam=len(trace),retained_steps=sum(x['accepted'] for x in trace),trace=trace,
        initial_old_surrogate=initial_old,final_old_surrogate=[float(A.outcome_surrogate(q,g,p).detach()) for g in old],
        no_Value_or_Critic=True,not_independent_fresh_rounds=True,no_projection_applied=True,unique_old_references=387)


def make(p):
    q=M.L.frozen(p,p['initial']['path']);q.requires_grad_(False);q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
    variables=[v for v in q.parameters() if v.requires_grad];assert sum(v.numel() for v in variables)==11906
    opt=torch.optim.Adam(variables,lr=p['fit_lr'],eps=p['Adam_eps'],foreach=False,fused=False)
    return q,opt
