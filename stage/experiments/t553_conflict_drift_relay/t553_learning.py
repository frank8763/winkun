"""Two-objective CAGrad-style coordination; no SciPy/offline installation.

Uses the author's 1-D convex objective with deterministic bounded golden search.
The second fixed mechanism also penalizes physical-time cumulative command drift.
No claim of continual-RL guarantee under Adam, old data or nonlinear dynamics.
"""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t552_joint_robust_night'));import t552_actor as A
P,G,torch,np=A.P,A.G,A.torch,A.np
read,save,sha,digest=A.read,A.save,A.sha,A.digest
Actor,frozen,make,state,load,fixed_hash,export,zero_export,feature_mean,location=A.Actor,A.frozen,A.make,A.state,A.load,A.fixed_hash,A.export,A.zero_export,A.feature_mean,A.location

def cagrad(grads,c=.5):
 assert grads.ndim==2 and grads.shape[0]==2 and torch.isfinite(grads).all();g1,g2=grads;aa=float(g1@g1);ab=float(g1@g2);bb=float(g2@g2);norm=math.sqrt(max(0.,(aa+2*ab+bb)/4));coeff=c*norm
 def objective(w):return .5*(w*(aa+ab)+(1-w)*(ab+bb))+coeff*math.sqrt(max(0.,w*w*aa+(1-w)**2*bb+2*w*(1-w)*ab)+1e-16)
 lo,hi=0.,1.;ratio=(math.sqrt(5)-1)/2;x=hi-ratio*(hi-lo);y=lo+ratio*(hi-lo);fx,fy=objective(x),objective(y)
 for _ in range(64):
  if fx<fy:hi,y,fy=y,x,fx;x=hi-ratio*(hi-lo);fx=objective(x)
  else:lo,x,fx=x,y,fy;y=lo+ratio*(hi-lo);fy=objective(y)
 w=min([0.,1.,(lo+hi)/2],key=objective);gw=w*g1+(1-w)*g2;lam=coeff/(float(gw.norm())+1e-8);direction=2*((g1+g2)/2+lam*gw)/(1+c*c)
 assert torch.isfinite(direction).all();return direction,dict(w=w,lambda_value=lam,gradient_norms=[math.sqrt(aa),math.sqrt(bb)],dot=ab,objective=objective(w),objective_at_equal=objective(.5),solver='GOLDEN64_PLUS_ENDPOINTS',not_GEM_constraint=True)

def cached_dataset(p,limit=None):
 data,pairs=A.cached_dataset(p,limit);by={r['job_id']:r for r in p['restore_rows']+p['keep_rows']+p['neighbor_rows']}
 for g in data:
  a=G.E.N.S.arrays(by[g['id']]['path']);mask=~a['ownership'].astype(bool) if g['kind']=='neighbor' else np.ones(len(a['states']),bool)
  if limit is not None:mask[limit:]=False
  g['dt']=torch.tensor(a['native_counts'][mask],dtype=torch.float64)*.002
 return data,pairs

def objectives(q,data,p):
 counts={k:sum(g['kind']==k for g in data) for k in ['restore','keep','neighbor']};values={k:0. for k in counts};losses={k:0. for k in counts};drift=0.
 for g in data:
  loc=location(q,g);err=P.action_loss(loc,g['target']).mean(-1) if g['kind']=='restore' else (loc-g['target']).square().mean(-1);v=(.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean())/counts[g['kind']];losses[g['kind']]=losses[g['kind']]+v;values[g['kind']]+=float(v.detach())
  if p['arm']=='integral' and g['kind']!='restore':
   # Commands, not impossible independent observation noise. Dimensionless
   # normalized time relative to 1 second; constant biases cost more over time.
   d=loc.clamp(-1,1)-g['target'].clamp(-1,1);dt=g['dt'];cum=(d*dt[:,None]).cumsum(0);v=(cum.square()/dt.cumsum(0).clamp_min(.002)[:,None]).mean()/counts[g['kind']];losses[g['kind']]=losses[g['kind']]+v;drift+=float(v.detach())
 values['physical_integral_drift']=drift;return [losses['restore'],losses['keep']+losses['neighbor']],values

def step(obj,losses,p,guard,event,k,label):
 q,opt=obj;guard();params=list(q.extra.parameters());vectors=[]
 for loss in losses:
  v=torch.autograd.grad(loss,params,retain_graph=True);vectors.append(torch.cat([x.reshape(-1) for x in v]))
 direction,meta=cagrad(torch.stack(vectors),p['cagrad_c']);opt.zero_grad(set_to_none=True);off=0
 for v in params:v.grad=direction[off:off+v.numel()].reshape_as(v).clone();off+=v.numel()
 norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True));event(label+'_BEFORE',k);opt.step();event(label+'_AFTER',k);return dict(norm=norm,coordination=meta)

def bc(obj,data,pairs,p,guard,event,steps=None):
 before=fixed_hash(obj[0]);trace=[]
 for g in obj[1].param_groups:g['lr']=p['bc_lr']
 for k in range(p['bc_steps'] if steps is None else steps):
  losses,v=objectives(obj[0],data,p);m=step(obj,losses,p,guard,event,k,'BC');trace.append(dict(step=k,loss=sum(float(x.detach()) for x in losses),**v,**m))
 assert fixed_hash(obj[0])==before;return trace

def rl_or_control(obj,arrays,data,pairs,p,guard,event):
 sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
 q,opt=obj;items,diag=Credit.peer_batch(obj,arrays,p);cached=[]
 for b in items:f,base=feature_mean(q,b['x']);cached.append(dict(b,f=f,base=base))
 fixed=fixed_hash(q);trace=[]
 for g in opt.param_groups:g['lr']=p['rl_lr']
 for k in range(p['rl_steps']):
  losses,v=objectives(q,data,p);policy=0.
  for b in cached:
   ratio=(P.L.density(location(q,b),b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();policy=policy-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(cached)
  total=[losses[0]+policy,losses[1]];m=step(obj,total,p,guard,event,k,'RL')
  with torch.no_grad():z=[P.L.density(location(q,b),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in cached];kl=sum(float((x.exp()-1-x).mean()) for x in z)/len(z)
  trace.append(dict(step=k,total_loss=sum(float(x.detach()) for x in total),policy_loss=float(policy.detach()),KL=kl,**v,**m))
  if kl>p['target_kl']:break
 assert fixed_hash(q)==fixed;return dict(trace=trace,peer_batch=diag,RL_Adam=len(trace),BC_control_Adam=0,new_Value=0)
