"""Matched reanchored replay with an ablation of pointwise old-output constraints.

Finite forward checks are empirical gates, not closed-loop safety certificates.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t614_feasible_recovery_step'))
import t614_learning as Previous
Prior,Old,P,G,torch,np=Previous.Prior,Previous.Old,Previous.P,Previous.G,Previous.torch,Previous.np
read,save,sha,digest=Previous.read,Previous.save,Previous.sha,Previous.digest
state,load,fixed,variables,frozen,export,make=Previous.state,Previous.load,Previous.fixed,Previous.variables,Previous.frozen,Previous.export,Previous.make
flat,unflat,project_halfspaces=Previous.flat,Previous.unflat,Previous.project_halfspaces
Credit=Previous.Credit

def dataset(p):
 data=Previous.dataset(p);q=frozen(p,p['student_initial']['path']);base=Old.initial(p)
 with torch.no_grad():
  for g in data:
   if g['kind']=='old' and g['id']==p['primary_new_task']:
    g['kind']='new_task'
   elif g['kind'] in ['old','recovery']:
    if g['kind']=='recovery':g['legacy_teacher_target']=g['target'].clone()
    if g['id']==p['primary_new_task']:
     g['kind']='new_task';g['source']='HISTORICAL_QUALIFIED_T550_GOAL'
    else:
     g['historical_target_sha']=digest(g['target']);g['target']=q.sequence(g['x']).detach();g['source']='CURRENT_RETAINED_FUNCTION_ANCHOR_ON_EXISTING_HISTORY'
  for row in p['current_success_rows']:
   assert row['success'] and not row['collision'] and row['policy_hash']==q.phash() and sha(row['path'])==row['sha256']
   a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);n=base.normalized(x);target=q.sequence(x).detach()
   assert float((target-torch.tensor(a['locations'])).abs().max())<1e-12
   data.append(dict(id=row['job_id'],kind='old',x=x,n=n,base=Old.Parent.Actor.sequence(base,x).detach(),reference=base.anchor(n[:,None])[0][:,0].detach(),target=target,mask=torch.ones(len(x),dtype=torch.bool),source='T615_CURRENT_RETAINED_ORIGINAL_START_SUCCESS'))
 assert len(data)==45 and sum(g['kind']=='old' for g in data)==40 and sum(g['kind']=='new_task' for g in data)==1
 assert sum(g['kind']=='correction' for g in data)==1 and sum(g['kind']=='recovery' for g in data)==3
 return data

def locations(pred,data,i):
 g=data[i];return (g['base']+pred[:len(g['n']),i]-g['reference'])[g['mask']]

def objective(q,data,packed,p):
 pred=q.branch(packed)[0];replay=[];new=[]
 for i,g in enumerate(data):
  loc=locations(pred,data,i);target=g['target'][g['mask']].clamp(-1,1);per=P.action_loss(loc,target).mean(-1)
  value=.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean()
  (new if g['kind'] in ['new_task','correction'] else replay).append(value)
 replay=torch.stack(replay);retention=.5*replay.mean()+.5*replay.topk(10).values.mean();gain=torch.stack(new).mean();loss=retention+gain
 return loss,dict(replay=float(retention.detach()),new_task_and_qualified_recovery=float(gain.detach()),KL=0.)

def protection(q,data,packed):return Previous.protection(q,data,packed)

def legacy_recovery_peak(q,data,packed):
 pred=q.branch(packed)[0]
 return max(float((locations(pred,data,i).clamp(-1,1)-g['legacy_teacher_target'][g['mask']].clamp(-1,1)).abs().max()) for i,g in enumerate(data) if g['kind']=='recovery')

def old_peak(q,data,packed):
 pred=q.branch(packed)[0];return max(float((locations(pred,data,i).clamp(-1,1)-g['target'][g['mask']].clamp(-1,1)).abs().max()) for i,g in enumerate(data) if g['kind']=='old')

def feasible_displacement(q,data,delta,p,guard):
 selected=[g for g in data if g['kind']=='recovery' or (p['mechanism']=='pointwise' and g['kind']=='old')]
 packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in selected]);named=list(q.branch.named_parameters());vs=tuple(v for name,v in named)
 def outputs(*values):
  pred=torch.func.functional_call(q.branch,{name:v for (name,_),v in zip(named,values)},(packed,))[0]
  return torch.cat([(locations(pred,selected,i).clamp(-1,1)-g['target'][g['mask']].clamp(-1,1)).reshape(-1) for i,g in enumerate(selected)])
 bounds=[];is_old=[]
 for g in selected:
  size=int(g['mask'].sum())*2;old=g['kind']=='old';bounds.extend([p['old_command_bound'] if old else p['protected_max_change']]*size);is_old.extend([old]*size)
 guard();e,jd=torch.autograd.functional.jvp(outputs,vs,unflat(delta,vs),create_graph=False,strict=False);e=e.detach();jd=jd.detach();bound=e.new_tensor(bounds);groups=torch.tensor(is_old,dtype=torch.bool)
 assert bool((e.abs()<=bound+1e-15).all());predicted=e+jd
 # Preserve T614's priority for the already tight recovery boundary.
 # Ranking only by predicted violation can omit a coordinate with microunits of slack.
 near=e.abs()>=bound-p['active_boundary_margin'];score=predicted.abs()-bound+near.to(e.dtype)*10.
 indices=[]
 for mask in [~groups,groups]:
  ix=torch.nonzero(mask).flatten()
  if len(ix):indices.extend(ix[torch.argsort(score[ix],descending=True,stable=True)[:p['projection_constraints_per_group']]].tolist())
 indices=torch.tensor(indices,dtype=torch.long);sign=torch.where(predicted[indices]>=0,1.,-1.);actual=outputs(*vs);normals=[]
 for k,index in enumerate(indices):
  guard();normals.append(flat(torch.autograd.grad(actual[index],vs,retain_graph=True))*sign[k])
 normals=torch.stack(normals).detach();rhs=(bound[indices]-sign*e[indices]).detach();assert bool((rhs>=0).all())
 d,info=project_halfspaces(delta,normals,rhs,p['projection_max_passes'],p['projection_tolerance'])
 # The finite active subset does not cover every time coordinate. Contract once
 # against both sides of every full-history JVP interval, with zero feasible.
 guard();_,jd_all=torch.autograd.functional.jvp(outputs,vs,unflat(d,vs),create_graph=False,strict=False);jd_all=jd_all.detach()
 moving=jd_all.abs()>1e-30;limits=torch.where(jd_all>=0,(bound-e),(bound+e))/jd_all.abs().clamp_min(1e-30)
 full_scale=min(1.,float(limits[moving].min())) if bool(moving.any()) else 1.;full_scale=max(0.,full_scale)*(1.-1e-12);d=d*full_scale
 full_violation=max(0.,float(((e+full_scale*jd_all).abs()-bound).max()));assert full_violation<=p['projection_tolerance']
 info.update(full_JVP_radial_scale=full_scale,full_JVP_max_violation=full_violation,full_linear_coordinates=len(e),both_interval_sides_checked=True)
 info.update(selected_indices=indices.tolist(),selected_old_constraints=int(groups[indices].sum()),selected_recovery_constraints=int((~groups[indices]).sum()),normals_sha256=digest(normals),rhs_sha256=digest(rhs),unprojected_L2=float(torch.linalg.vector_norm(delta)),projected_L2=float(torch.linalg.vector_norm(d)),all_nonlinear_coordinates_must_pass=True)
 return d,info

def fit(obj,data,p,guard,event,arrays=None,steps=None):
 assert arrays is None and p['arm']=='command' and p['mechanism'] in ['reanchor','pointwise'];q,opt=obj;body=fixed(q);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);before=Old.measure(q,data);trace=[];applied=0
 names=[name for name,v in q.named_parameters() if name.startswith('branch.')]
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed,p);value=float(loss.detach());loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj)
  delta=flat([proposal['actor'][name]-prior['actor'][name] for name in names])
  with torch.no_grad():
   for name,v in q.named_parameters():
    if name.startswith('branch.'):v.copy_(prior['actor'][name])
  projected,diag=feasible_displacement(q,data,delta,p,guard);trials=[];accepted=False
  for factor in p['step_factors']:
   guard()
   with torch.no_grad():
    for name,d in zip(names,unflat(projected,variables(q))):dict(q.named_parameters())[name].copy_(prior['actor'][name]+factor*d)
    nxt,detail=objective(q,data,packed,p);raw,cmd=protection(q,data,packed);peak=old_peak(q,data,packed);legacy=legacy_recovery_peak(q,data,packed)
   okay=diag['converged'] and diag['projected_L2']>1e-14 and math.isfinite(float(nxt)) and float(nxt)<=value+1e-12 and cmd<=p['protected_max_change'] and (p['mechanism']=='reanchor' or peak<=p['old_command_bound'])
   trials.append(dict(factor=factor,loss=float(nxt),raw_max=raw,command_max=cmd,old_command_max=peak,legacy_teacher_recovery_command_max_diagnostic=legacy,parts=detail,accepted=okay))
   if okay:accepted=True;applied+=1;break
  if not accepted:load(obj,prior)
  trace.append(dict(step=k,before_loss=value,before_parts=parts,gradient_norm=norm,projection=diag,trials=trials,applied=accepted))
  if not accepted:break
 assert fixed(q)==body
 return dict(attempted_Adam=len(trace),applied_parameter_steps=applied,trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,new_RL_attempted_Adam=0,new_BC_attempted_Adam=len(trace),new_Critic=0,mechanism=p['mechanism'],full_history_BPTT=True,reanchored_old_histories=40,reanchored_recovery_histories=3,recovery_bound_relative_to_retained_parent_not_legacy_teacher=True,qualified_goal_supervision_histories=2,teacher_hidden_never_used=True,scaled_full_proposal_Adam_moments=True,reject_restores_entire_ActorAdam=True,not_stability_or_safety_proof=True)
