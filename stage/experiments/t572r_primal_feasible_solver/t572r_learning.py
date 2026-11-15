"""Same convex output-head problem, primal-feasible working-set QP recovery.

No ADMM continuation or tolerance change. Starts strictly feasible at zero.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t572_functional_trust_region'));import t572_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
make,fixed,state,load,frozen,export,dataset,matrices,measure,apply=Prior.make,Prior.fixed,Prior.state,Prior.load,Prior.frozen,Prior.export,Prior.dataset,Prior.matrices,Prior.measure,Prior.apply

def one(Q,c,A,b,p,guard):
 # Standard primal-feasible active set. Cx<=d contains upper and lower rows.
 C=torch.cat([A,-A]);d=torch.cat([b,b]);D=len(Q);N=len(A);factor=torch.linalg.cholesky(Q);Qi=torch.cholesky_solve(torch.eye(D,dtype=Q.dtype),factor);x=torch.zeros(D,dtype=Q.dtype);active=[];trace=[];multipliers=torch.zeros(len(C),dtype=Q.dtype);done=False;lin_solves=1
 for k in range(p['working_set_max_iterations']):
  guard();grad=Q@x+c;free=-Qi@grad
  if active:
   B=C[active];K=B@Qi@B.T;rhs=B@free
   # Linear independence is required; no regularization changing the QP.
   eig=torch.linalg.eigvalsh(K);assert float(eig.min())>p['active_rank_tolerance']*max(float(eig.max()),1e-30),'dependent working set'
   mu=torch.linalg.solve(K,rhs);direction=free-Qi@B.T@mu;lin_solves+=1
  else:mu=torch.zeros(0,dtype=Q.dtype);direction=free
  assert torch.isfinite(direction).all()
  if float(direction.norm())<=p['direction_zero_tolerance']:
   if not active or float(mu.min())>=-p['kkt_tolerance']:
    multipliers.zero_()
    if active:multipliers[active]=mu
    done=True;trace.append([k+1,len(active),0.,float((C@x-d).max()),float(mu.min()) if active else 0.]);break
   # Remove most negative multiplier; ties use fixed active order.
   j=int(mu.argmin());removed=active.pop(j);trace.append([k+1,len(active),-1.,float((C@x-d).max()),float(mu[j])]);continue
  v=C@direction;slack=d-C@x;eligible=v>p['blocking_direction_tolerance']
  if active:eligible[active]=False
  ratio=torch.where(eligible,slack.clamp(min=0)/v.clamp(min=1e-300),torch.full_like(v,float('inf')));a,idx=ratio.min(0);alpha=min(1.,float(a));assert alpha>=0;x=x+alpha*direction
  assert float((C@x-d).max())<=p['kkt_tolerance'],'primal feasibility lost'
  if alpha<1.-1e-12:
   blocker=int(idx);assert blocker not in active;active.append(blocker);assert len(active)<=D
  trace.append([k+1,len(active),alpha,float((C@x-d).max()),float(direction.norm())])
 if not done:
  # Use only final working-set multipliers for diagnostic, never invent a pass.
  grad=Q@x+c
  if active:
   B=C[active];mu=torch.linalg.solve(B@Qi@B.T,-B@Qi@grad);multipliers[active]=mu;lin_solves+=1
 dual=(multipliers[:N]-multipliers[N:]);cert=Prior.certificate(Q,c[:,None],A,b[:,None],x[:,None],dual[:,None]);cert['multiplier_negative_violation']=float((-multipliers).clamp(min=0).max())
 return x,dual,dict(iterations=k+1,working_set_converged=done,working_set=active,linear_solves=lin_solves,certificate=cert,trace=trace)

def solve(Q,c,rawA,norm,arm,p,guard):
 assert arm=='bounded';A=rawA/norm[:,None];b=torch.full((len(A),),p['drift_limit'],dtype=Q.dtype)/norm;xs=[];ys=[];meta=[]
 for i in range(2):
  x,y,m=one(Q,c[:,i],A,b,p,guard);xs.append(x);ys.append(y);meta.append(m)
 w=torch.stack(xs,1);lam=torch.stack(ys,1);cert=Prior.certificate(Q,c,A,b[:,None],w,lam);dual_violation=max(m['certificate']['multiplier_negative_violation'] for m in meta);raw=float((rawA@w).abs().max());passed=all(m['working_set_converged'] for m in meta) and cert['finite'] and max(cert['primal_inf'],cert['stationarity_inf'],cert['complementarity_inf'],dual_violation)<=p['kkt_tolerance'] and raw<=p['drift_limit']+p['raw_drift_tolerance']
 info=dict(arm='bounded',method='PRIMAL_FEASIBLE_WORKING_SET_ZERO_START',iterations=sum(m['iterations'] for m in meta),max_iterations_per_output=p['working_set_max_iterations'],closed_form_linear_solves=sum(m['linear_solves'] for m in meta),certificate_pass=passed,certificate=dict(cert,multiplier_negative_violation=dual_violation),raw_memory_location_drift_max=raw,coefficient_norm=float(w.norm()),objective=float(.5*(w*(Q@w)).sum()+(c*w).sum()),per_output=meta,new_Adam=0,new_RL=0)
 assert torch.isfinite(w).all();return w,lam,info
