"""GEM-inspired linear recovery-output constraints on the actual Adam displacement.
Finite nonlinear forward gates still decide acceptance; no safety certificate.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t613_qualified_recovery_learning'))
import t613_learning as Previous
Prior,Old,P,G,torch,np=Previous.Prior,Previous.Old,Previous.P,Previous.G,Previous.torch,Previous.np
read,save,sha,digest=Previous.read,Previous.save,Previous.sha,Previous.digest
state,load,fixed,variables,frozen,export,make,dataset,objective,protection=Previous.state,Previous.load,Previous.fixed,Previous.variables,Previous.frozen,Previous.export,Previous.make,Previous.dataset,Previous.objective,Previous.protection
Credit=Previous.Credit

def flat(vs):return torch.cat([v.reshape(-1) for v in vs])
def unflat(z,vs):
 out=[];off=0
 for v in vs:n=v.numel();out.append(z[off:off+n].reshape_as(v));off+=n
 assert off==len(z);return tuple(out)

def project_halfspaces(delta,normal,rhs,max_passes,tolerance):
 norm=torch.linalg.vector_norm(normal,dim=1);keep=norm>1e-14
 if not bool(keep.all()):
  assert bool((rhs[~keep]>=-tolerance).all());normal=normal[keep];rhs=rhs[keep];norm=norm[keep]
 if not len(norm):return delta.clone(),dict(passes=0,converged=True,max_violation=0.)
 normal=normal/norm[:,None];rhs=rhs/norm;x=delta.clone();correction=torch.zeros_like(normal)
 for it in range(max_passes):
  before=x.clone()
  for i in range(len(normal)):
   y=x+correction[i];n=normal[i];z=y-torch.clamp(torch.dot(n,y)-rhs[i],min=0)*n;correction[i]=y-z;x=z
  violation=max(0.,float((normal@x-rhs).max()))
  if violation<=tolerance and float((x-before).abs().max())<=tolerance:break
 dykstra_violation=violation;dot=normal@x;bad=dot>rhs;contraction=1.
 if bool(bad.any()):
  assert bool((rhs>=0).all()),'Zero displacement must be feasible for radial repair'
  contraction=max(0.,min(1.,float((rhs[bad]/dot[bad]).min())))*(1.-1e-12);x=x*contraction
 violation=max(0.,float((normal@x-rhs).max()))
 return x,dict(passes=it+1,converged=violation<=tolerance,max_violation=violation,dykstra_max_violation_before_radial_repair=dykstra_violation,radial_contraction=contraction,finite_approximate_not_exact_Euclidean_projection=True)

def feasible_displacement(q,data,delta,p,guard):
 rec=[g for g in data if g['kind']=='recovery'];assert len(rec)==3;packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in rec]);named=list(q.branch.named_parameters());vs=tuple(v for name,v in named)
 def outputs(*values):
  pred=torch.func.functional_call(q.branch,{name:v for (name,_),v in zip(named,values)},(packed,))[0];out=[]
  for i,g in enumerate(rec):
   loc=(g['base']+pred[:len(g['n']),i]-g['reference'])[g['mask']];out.append((loc.clamp(-1,1)-g['target'][g['mask']].clamp(-1,1)).reshape(-1))
  return torch.cat(out)
 guard();e,jd=torch.autograd.functional.jvp(outputs,vs,unflat(delta,vs),create_graph=False,strict=False);e=e.detach();jd=jd.detach();bound=p['protected_max_change'];assert float(e.abs().max())<=bound
 predicted=e+jd;violation=predicted.abs()-bound;near=e.abs()>=bound-p['active_boundary_margin'];score=violation+near.to(e.dtype)*10.;indices=torch.argsort(score,descending=True,stable=True)[:min(p['projection_max_constraints'],len(e))];sign=torch.where(predicted[indices]>=0,1.,-1.)
 actual=outputs(*vs);normal=[]
 for k,index in enumerate(indices):
  guard();normal.append(flat(torch.autograd.grad(actual[index],vs,retain_graph=True))*sign[k])
 normal=torch.stack(normal).detach();rhs=(bound-sign*e[indices]).detach();assert torch.isfinite(normal).all() and torch.isfinite(rhs).all()
 d,info=project_halfspaces(delta,normal,rhs,p['projection_max_passes'],p['projection_tolerance']);info.update(selected_indices=indices.tolist(),selected_normals_sha256=digest(normal),selected_rhs_sha256=digest(rhs),initial_recovery_command_max=float(e.abs().max()),unprojected_linear_max_violation=max(0.,float(violation.max())),unprojected_L2=float(torch.linalg.vector_norm(delta)),projected_L2=float(torch.linalg.vector_norm(d)),only_selected_linear_constraints_NOT_full_nonlinear_guarantee=True)
 return d,info

def fit(obj,data,p,guard,event,arrays=None,steps=None):
 assert arrays is None and p['arm']=='command';q,opt=obj;body=fixed(q);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);before=Old.measure(q,data);trace=[];applied=0
 names=[name for name,v in q.named_parameters() if name.startswith('branch.')];vs=variables(q)
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed,p);value=float(loss.detach());loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(vs,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj);delta=flat([proposal['actor'][name]-prior['actor'][name] for name in names])
  with torch.no_grad():
   for name,v in q.named_parameters():
    if name.startswith('branch.'):v.copy_(prior['actor'][name])
  projected,diag=feasible_displacement(q,data,delta,p,guard);trials=[];accepted=False
  for factor in p['step_factors']:
   guard()
   with torch.no_grad():
    for name,d in zip(names,unflat(projected,vs)):dict(q.named_parameters())[name].copy_(prior['actor'][name]+factor*d)
    nxt,detail=objective(q,data,packed,p);raw,cmd=protection(q,data,packed)
   okay=diag['converged'] and math.isfinite(float(nxt)) and float(nxt)<=value+1e-12 and cmd<=p['protected_max_change']
   trials.append(dict(factor=factor,loss=float(nxt),raw_max=raw,command_max=cmd,parts=detail,accepted=okay))
   if okay:accepted=True;applied+=1;break
  if not accepted:load(obj,prior)
  trace.append(dict(step=k,before_loss=value,before_parts=parts,gradient_norm=norm,projection=diag,trials=trials,applied=accepted))
  if not accepted:break
 assert fixed(q)==body
 return dict(attempted_Adam=len(trace),applied_parameter_steps=applied,trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,new_RL_attempted_Adam=0,new_BC_attempted_Adam=len(trace),new_Critic=0,full_history_BPTT=True,actual_Adam_displacement_projected=True,teacher_suffix_mask_unchanged=True,scaled_Adam_moments_from_full_proposal=True,rejected_entire_step_restored=True,not_standard_GEM_or_safety_proof=True)

def fit(obj,data,p,guard,event,arrays=None,steps=None):
 assert arrays is None and p['arm']=='command';q,opt=obj;body=fixed(q);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);before=Old.measure(q,data);trace=[];applied=0
 names=[name for name,v in q.named_parameters() if name.startswith('branch.')]
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed,p);value=float(loss.detach());loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj)
  delta=flat([proposal['actor'][name]-prior['actor'][name] for name in names])
  # Keep exactly this proposal's Adam moments; linearize the output at before weights.
  with torch.no_grad():
   for name,v in q.named_parameters():
    if name.startswith('branch.'):v.copy_(prior['actor'][name])
  projected,projection=feasible_displacement(q,data,delta,p,guard);trials=[];accepted=False
  for factor in p['step_factors']:
   guard()
   with torch.no_grad():
    for name,d in zip(names,unflat(projected,variables(q))):dict(q.named_parameters())[name].copy_(prior['actor'][name]+factor*d)
    nxt,detail=objective(q,data,packed,p);raw,cmd=protection(q,data,packed)
   okay=projection['converged'] and projection['projected_L2']>1e-14 and math.isfinite(float(nxt)) and float(nxt)<=value+1e-12 and cmd<=p['protected_max_change']
   trials.append(dict(factor=factor,loss=float(nxt),raw_max=raw,command_max=cmd,parts=detail,accepted=okay))
   if okay:accepted=True;applied+=1;break
  if not accepted:load(obj,prior)
  trace.append(dict(step=k,before_loss=value,before_parts=parts,gradient_norm=norm,projection=projection,trials=trials,applied=accepted))
  if not accepted:break
 assert fixed(q)==body
 return dict(attempted_Adam=len(trace),applied_parameter_steps=applied,trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,new_RL_attempted_Adam=0,new_BC_attempted_Adam=len(trace),new_Critic=0,full_history_BPTT=True,actual_Adam_displacement_projected=True,teacher_suffix_mask_unchanged=True,scaled_Adam_moments_from_full_proposal=True,rejected_entire_step_restored=True,not_standard_GEM_or_safety_proof=True)
