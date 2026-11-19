"""Atomic finite-block constrained replay: actual Actor+Adam commits or rollback.

No projected gradient safety claim. Fixed parent commands, causal full history.
"""
from pathlib import Path
import copy,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t647_joint_success_coverage'))
import t647_learning as Prior
Base=Prior.Base
L,P,E,torch,np=Prior.L,Prior.P,Prior.E,Prior.torch,Prior.np
state,load,export,make=Prior.state,Prior.load,Prior.export,Prior.make
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
schedule=Prior.schedule


def response_gate(metrics,p):
    return bool(all(v['peak_command_error']<=p['response_peak'] and v['RMS_command_error']<=p['response_rms'] for v in metrics))


@torch.no_grad()
def responses(q,groups,p,guard):
    result=[];device=next(q.parameters()).device
    for off in range(0,len(groups),p['response_batch']):
        guard();items=groups[off:off+p['response_batch']]
        prediction=Base.evaluate_sequences(q,items,device)
        for loc,g in zip(prediction,items):
            error=loc.clamp(-1,1)-g['target'].to(device).clamp(-1,1)
            result.append(dict(id=g['id'],peak_command_error=float(error.abs().max()),RMS_command_error=float(error.square().mean().sqrt())))
    return result


@torch.no_grad()
def validation(q,groups,p,guard):
    values={};device=next(q.parameters()).device
    for off in range(0,len(groups),p['response_batch']):
        guard();items=groups[off:off+p['response_batch']]
        prediction=Base.evaluate_sequences(q,items,device)
        for loc,g in zip(prediction,items):
            v=P.action_loss(loc,g['target'].to(device)).mean()
            values.setdefault(g['id'],[]).append(float(v))
    return dict(task_mean_loss={key:sum(v)/len(v) for key,v in values.items()},task_equal_mean_loss=sum(sum(v)/len(v) for v in values.values())/len(values),heldout_trajectories=len(groups),not_new_geometry=True)


def commit_scaled(obj,before,proposed,factor):
    """Finite proposal scaling; retained moments are the full-block proposal's.

    This is explicitly a custom constrained-Adam commit, not standard Adam.
    Factor zero restores ALL before Actor+Adam. Factor one keeps proposed bytes.
    """
    if factor==0:
        load(obj,before);return
    load(obj,proposed)
    if factor!=1:
        with torch.no_grad():
            blended={key:before['actor'][key]+factor*(value-before['actor'][key]) if value.is_floating_point() else value for key,value in proposed['actor'].items()}
            obj[0].load_state_dict(blended)


def metrics_summary(metrics,p):
    return dict(all384_response_digest=digest(metrics),
        max_peak_command_error=max(v['peak_command_error'] for v in metrics),
        max_task_RMS_command_error=max(v['RMS_command_error'] for v in metrics),
        violating_ids=[v['id'] for v in metrics if v['peak_command_error']>p['response_peak'] or v['RMS_command_error']>p['response_rms']],
        passed=response_gate(metrics,p))


def fit(obj,old,new,p,guard,event):
    q,opt=obj;trace=[];blocks=[];assert len(old)==384 and len(new)==p['training_trajectory_count']
    assert p['fit_steps']%p['commit_block']==0
    initial=digest(state(obj));committed=0
    for start in range(0,p['fit_steps'],p['commit_block']):
        before=state(obj);before_hash=digest(before)
        for k in range(start,start+p['commit_block']):
            guard();a,b=schedule(old,new,k,p);opt.zero_grad(set_to_none=True)
            loss,repair,keep=Base.objective(q,a,b,dict(p,arm='full_command'));loss.backward()
            variables=[v for v in q.parameters() if v.requires_grad]
            norm=float(torch.nn.utils.clip_grad_norm_(variables,1.,error_if_nonfinite=True))
            event('BEFORE',k);opt.step();event('AFTER',k)
            assert all(torch.isfinite(v).all() for v in variables)
            trace.append(dict(step=k,old_ids=[g['id'] for g in a],new_sources=[g['source_sha256'] for g in b],loss=float(loss.detach()),repair=float(repair.detach()),keep=float(keep.detach()),norm=norm))
        # Measure all four actual parameter displacements in both arms, with
        # no extra optimizer calls. Always relative to fixed T501 responses.
        proposed=state(obj);proposal_hash=digest(proposed);candidates=[]
        for factor in p['candidate_scales']:
            commit_scaled(obj,before,proposed,factor)
            metrics=responses(q,old,p,guard)
            candidates.append(dict(scale=factor,**metrics_summary(metrics,p)))
        chosen=1. if p['arm']=='replay' else next((v['scale'] for v in candidates if v['passed']),0.)
        commit_scaled(obj,before,proposed,chosen)
        if chosen:committed+=p['commit_block']
        else:assert digest(state(obj))==before_hash
        blocks.append(dict(first_step=start,proposals=p['commit_block'],before_state_hash=before_hash,
            proposed_state_hash=proposal_hash,retained_state_hash=digest(state(obj)),
            accepted_scale=chosen,committed=bool(chosen),four_actual_response_checks=candidates))
    assert len(trace)==p['fit_steps']
    if p['arm']=='constrained':assert response_gate(responses(q,old,p,guard),p)
    return dict(Actor_Adam=len(trace),committed_Actor_Adam=committed,arm=p['arm'],new_tasks=8,
       training_trajectories=len(new),old_tasks=384,full_causal_history=True,initial_state_hash=initial,
       response_checks=len(blocks)*len(p['candidate_scales']),block_commit_size=p['commit_block'],trace=trace,blocks=blocks)
