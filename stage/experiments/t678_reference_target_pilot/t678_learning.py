"""One real step per target arm; same task-balanced episodic objective."""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t677_reference_target_contract'))
import t677_analysis as C
sys.path.insert(0,str(ROOT/'stage/experiments/t671_joint_outcome_feasibility'))
import t671_learning as Prior
B,M,T,torch,np=C.B,C.M,C.T,C.torch,C.np
read,sha,digest=C.read,C.sha,C.digest
make,optimizer_state,load=Prior.make,Prior.optimizer_state,Prior.load

def data(p,q,guard):
    rows=B.dataset(p);assert len(rows)==407
    refs=M.references(p,q,guard)
    if p['target_arm']=='self':refs=C.own_targets(q,refs)
    else:assert p['target_arm']=='executed'
    return rows,None,M.pack(refs)

def fit(p,q,opt,rows,unused,keep,guard,event):
    assert p['Actor_steps']==1 and not opt.state
    variables=[v for v in q.parameters() if v.requires_grad];fixed=M.L.fixed_hash(q)
    before=optimizer_state(q,opt);theta=torch.cat([v.detach().flatten().clone() for v in variables])
    opt.zero_grad(set_to_none=True);pg=0.;panel={}
    for key,rr in rows.items():
        guard();g,y=B.A.group(q,rr,p);term=Prior.pg(q,g,p)/407;term.backward();pg+=float(term.detach())
        if key in p['gradient_panel_ids']:
            with torch.no_grad():
                for item in g:item['initial']=M.location(q,item).detach().clone()
            panel[key]=g
    kloss,losses=M.retention(q,keep,False);(p['replay_weight']*kloss).backward()
    actual=torch.cat([v.grad.detach().flatten() for v in variables]).numpy().copy()
    with np.load(p['prior_target_gradients'],allow_pickle=False) as f:
        refgrad=f[p['target_arm']+'_retention_gradient'];expected_delta=f[p['target_arm']+'_formula_displacement']
    with np.load(p['outcome_gradients'],allow_pickle=False) as f:expected=-f['task_gradients'].mean(0)+p['replay_weight']*refgrad
    gradient_error=float(np.max(np.abs(actual-expected)));assert gradient_error<1e-10
    norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
    clipped=torch.cat([v.grad.detach().flatten() for v in variables]).numpy().copy()
    event('BEFORE',0);opt.step();event('AFTER',0)
    delta=(torch.cat([v.detach().flatten() for v in variables])-theta).numpy().copy()
    formula=-p['fit_lr']*clipped/(np.abs(clipped)+p['Adam_eps'])
    formula_error=float(np.max(np.abs(delta-formula)));prior_error=float(np.max(np.abs(delta-expected_delta)))
    assert formula_error<1e-12 and prior_error<1e-12
    values={key:B.metrics(q,g,p) for key,g in panel.items()}
    finite=all(bool(torch.isfinite(v).all()) for v in variables)
    maxkl=max(x['task_episode_mean_KL'] for x in values.values())
    accepted=finite and maxkl<=p['max_task_latent_KL']
    # A rejected proposed checkpoint is preserved by the runtime as well.
    proposed=optimizer_state(q,opt)
    if not accepted:
        load(q,opt,before);assert digest(optimizer_state(q,opt))==digest(before)
    assert M.L.fixed_hash(q)==fixed
    info=dict(Actor_Adam=1,retained_steps=int(accepted),target_arm=p['target_arm'],same407_terminal_objective=True,
              no_failed_C_reward=True,pg=pg,keep_before=float(kloss.detach()),reference_worst_before=float(losses.max().detach()),
              gradient_norm_before_clip=norm,gradient_error=gradient_error,first_Adam_formula_error=formula_error,
              prior_registered_delta_error=prior_error,actual_displacement_hash=digest(delta),actual_displacement_norm=float(np.linalg.norm(delta)),
              max_actual_sigma_task_mean_KL=maxkl,finite=finite,accepted=accepted,
              old_nonlinear_negative=[key for key,v in values.items() if key in p['before_success_ids'] and v['sample_surrogate']<-1e-12],
              nonlinear_panel=values,old_surrogate_diagnostics_not_safety_acceptance=True,new_Value=0)
    return info,proposed
