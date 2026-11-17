"""Conflict-aware task PG; identical losses/data, meanSGD versus CAGradSGD."""
from pathlib import Path
import sys,os,time,copy,math
REQUESTED_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES','')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_learning as P
os.environ['CUDA_VISIBLE_DEVICES']=REQUESTED_CUDA
L,torch,np=P.L,P.torch,P.np
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t627_conflict_aware_policy';DOC=ROOT/'stage/docs/day_20261117/t627'
read,save,sha,digest=P.read,P.save,P.sha,P.digest
configure_gpu=P.configure_gpu;batch=P.batch;memories=P.memories;trust=P.trust;state=P.state;load=P.load
def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);return c
def make(p):
 q,_=P.make(p);return q,torch.optim.SGD(L.variables(q),lr=p['actor_lr'],momentum=0.,foreach=False)
def simplex(v):
 u=np.sort(v)[::-1];css=np.cumsum(u)-1.;rho=np.nonzero(u-css/(np.arange(len(v))+1)>0)[0][-1];return np.maximum(v-css[rho]/(rho+1),0.)
def combine(grads,p,mode):
 assert grads.ndim==2 and grads.shape[1]>=2 and torch.isfinite(grads).all();mean=grads.mean(1);gram=(grads.T@grads).detach().cpu().numpy();scale=max(float(np.trace(gram)),1e-30);A=gram/scale;n=len(A);b=np.full(n,1/n);c=p['cagrad_c'];coefficient=c*math.sqrt(max(float(A.mean()),0.)+1e-8)
 if mode=='mean':direction=mean;weights=b;info=dict(dual_iterations=0,dual_gap=None,solver='not_needed')
 else:
  w=b.copy();Ab=A@b
  def objective(v):return float(v@Ab+coefficient*math.sqrt(max(float(v@A@v),0.)+1e-8))
  step=1.;converged=False
  for it in range(p['dual_max_iterations']):
   Aw=A@w;gradient=Ab+coefficient*Aw/math.sqrt(max(float(w@Aw),0.)+1e-8);gap=float(w@gradient-gradient.min())
   if gap<=p['dual_tolerance']:converged=True;break
   value=objective(w)
   for _ in range(p['dual_backtrack']):
    new=simplex(w-step*gradient);delta=new-w
    if objective(new)<=value+float(gradient@delta)+float(delta@delta)/(2*step)+1e-15:break
    step*=.5
   else:raise AssertionError('CAGrad dual backtrack failed')
   w=new;step=min(step*1.2,1000.)
  assert converged,('CAGrad dual not converged',gap)
  weights=w;gw=grads@grads.new_tensor(w);coef=c*math.sqrt(max(float(gram.mean()),0.)+1e-8*scale);den=math.sqrt(max(float(w@gram@w),0.)+1e-8*scale);direction=(mean+(coef/(den+1e-8*math.sqrt(scale)))*gw)/(1+c*c)
  info=dict(dual_iterations=it,dual_gap=gap,dual_objective=objective(w),solver='simplex_projected_gradient_same_convex_CAGrad_dual',regularization_scaled_epsilon=1e-8,original_official_scipy_unavailable=True)
 dots=(grads.T@direction).detach().cpu().tolist();norms=torch.linalg.vector_norm(grads,dim=0);cos=(grads.T@grads/(norms[:,None]*norms[None,:]+1e-30)).detach().cpu().numpy();info.update(weights=weights.tolist(),task_gradient_dot_descent_direction=dots,negative_local_task_directions=sum(x<0 for x in dots),negative_pair_cosines=int((cos[np.triu_indices(n,1)]<0).sum()),gradient_gram_sha=digest(gram),combined_norm=float(torch.linalg.vector_norm(direction)),only_local_sample_gradient_NOT_closed_loop_guarantee=True)
 return direction,info
def task_gradients(q,items,refs,p):
 params=L.variables(q);allg=[];values=[]
 # Same old-task function penalty contributes equally to each task objective.
 q.zero_grad(set_to_none=True);keep=next(q.branch.parameters()).new_zeros(())
 for r in refs:
  loc=q.sequence(r['x'])[r['mask']];target=r['target'][r['mask']];keep=keep+.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean()/len(refs)
 kg=torch.autograd.grad(p['keep_weight']*keep,params);flatkeep=torch.cat([v.flatten() for v in kg]).detach()
 for group in items:
  loss=next(q.branch.parameters()).new_zeros(())
  for b in group:
   loc=q.sequence(b['x']);z=L.P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old'];ratio=z.exp();loss=loss-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/4
  g=torch.autograd.grad(loss,params);allg.append(torch.cat([v.flatten() for v in g]).detach()+flatkeep);values.append(float(loss.detach()))
 return torch.stack(allg,1),dict(task_policy_losses=values,keep=float(keep.detach()),keep_gradient_norm=float(torch.linalg.vector_norm(flatkeep)))
def fit(obj,groups,p,guard,event,steps=None):
 q,opt=obj;body=L.fixed(q);items,diag=batch(q,groups,p);refs=memories(q,p);trace=[];applied=0
 for k in range(p['actor_steps'] if steps is None else steps):
  guard();before=state(obj);grads,detail=task_gradients(q,items,refs,p);direction,combination=combine(grads,p,p['arm']);opt.zero_grad(set_to_none=True);off=0
  for v in L.variables(q):v.grad=direction[off:off+v.numel()].reshape_as(v).clone();off+=v.numel()
  assert off==len(direction);norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);kl=trust(q,items,p);okay=max(kl)<=p['max_task_latent_KL'] and np.isfinite(kl).all()
  if okay:applied+=1
  else:load(obj,before)
  displacement=torch.cat([(v-before['actor'][name]).detach().flatten() for name,v in q.named_parameters() if name.startswith('branch.')]);actual_predicted_change=(grads.T@displacement).detach().cpu().tolist()
  trace.append(dict(step=k,gradient_norm=norm,parameter_displacement_L2=float(torch.linalg.vector_norm(displacement)),actual_task_first_order_loss_changes=actual_predicted_change,task_latent_KL=kl,accepted=bool(okay),**detail,combination=combination))
  if not okay:break
 assert L.fixed(q)==body
 return dict(attempted_optimizer=len(trace),applied=applied,trace=trace,peer_batch=diag,optimizer='SGD_no_momentum',new_Critic=0,actual_task_gradients=True,full_current_history_BPTT=True,not_standard_GEM_CPO_or_safety_certificate=True)
