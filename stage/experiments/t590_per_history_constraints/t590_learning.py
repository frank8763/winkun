"""Same function-preserving Actor, finite-history constrained vs fixed-penalty BC.

No RL claims: the only new algorithm state is39nonnegative dual variables.
"""
from pathlib import Path
import sys,copy,gzip,io
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t586_function_preserving_initialization'))
import t586_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
feature,loc,fixed,frozen,fresh=Prior.feature,Prior.loc,Prior.fixed,Prior.frozen,Prior.fresh

def make(p):
 q,opt=Prior.make(p);q.keep_duals=torch.zeros(39,dtype=torch.float64)
 return q,opt
def state(obj):
 s=Prior.state(obj);s['keep_duals']=obj[0].keep_duals.clone();return s
def load(obj,s):
 v=copy.deepcopy(s);obj[0].keep_duals=v.pop('keep_duals').clone();Prior.load(obj,v)

def dataset(p):
 q=fresh(p);keep=[];teach=[]
 for r in p['keep_rows']:
  assert sha(r['path'])==r['sha256'];a=G.E.N.S.arrays(r['path']);f,b=feature(q,torch.tensor(a['states']));keep.append(dict(id=r['job_id'],f=f,b=b,target=b.clone()))
 for r in p['recovery_rows']:
  assert sha(r['path'])==r['sha256'] and r['success'] and not r['collision'];a=G.E.N.S.arrays(r['path']);k=int(a['teacher_first_declared']);assert a['ownership'][k:].all()
  f,b=feature(q,torch.tensor(a['states']));teach.append(dict(id=r['job_id'],f=f[k:],b=b[k:],target=torch.tensor(a['actions'][k:],dtype=torch.float64)))
 assert len(keep)==len({g['id'] for g in keep})==39 and len(teach)==3
 return keep,teach

def metrics(q,data):
 with torch.no_grad():
  keep,teach=data;err=[(loc(q,g['f'],g['b'])-g['target']).abs() for g in keep]
  return dict(teacher=[dict(command_MAE=float((loc(q,g['f'],g['b']).clamp(-1,1)-g['target']).abs().mean()),command_max=float((loc(q,g['f'],g['b']).clamp(-1,1)-g['target']).abs().max())) for g in teach],keep_max_by_history=[float(e.max()) for e in err],keep_location_MAE=float(torch.cat([e.flatten() for e in err]).mean()))

def merged(groups):
 return dict(f=torch.cat([g['f'] for g in groups]),b=torch.cat([g['b'] for g in groups]),target=torch.cat([g['target'] for g in groups]),weights=torch.cat([torch.full((len(g['f']),),1/(len(groups)*len(g['f'])),dtype=torch.float64) for g in groups]),lengths=[len(g['f']) for g in groups])

def fit(obj,data,p,guard,event,steps=None):
 q,opt=obj;before=metrics(q,data);body=fixed(q);km,tm=map(merged,data);trace=[]
 for k in range(p['BC_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True)
  tv=loc(q,tm['f'],tm['b']);tl=(.5*((tv-tm['target'])/.2).square().mean(-1)*tm['weights']).sum()
  err=loc(q,km['f'],km['b'])-km['target'];mx=torch.stack([v.abs().max() for v in err.split(km['lengths'])]);g=(mx/p['keep_max_tolerance']).square()-1
  if p['method']=='fixed_penalty':
   kl=p['keep_weight']*(.5*(err/.2).square().mean(-1)*km['weights']).sum()
  else:
   assert p['method']=='per_history_constraints';rho=p['dual_rho'];kl=((torch.relu(q.keep_duals+rho*g).square()-q.keep_duals.square())/(2*rho)).mean()
  (tl+kl).backward();norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step()
  with torch.no_grad():
   # Dual uses the updated nonlinear policy, not the pre-update surrogate.
   if p['method']=='per_history_constraints':
    e=loc(q,km['f'],km['b'])-km['target'];newg=(torch.stack([v.abs().max() for v in e.split(km['lengths'])])/p['keep_max_tolerance']).square()-1
    q.keep_duals=torch.relu(q.keep_duals+p['dual_rho']*newg)
   assert torch.isfinite(q.keep_duals).all()
  event('AFTER',k);trace.append([float(tl.detach()),float(kl.detach()),norm,float(mx.detach().max()),float(q.keep_duals.max())])
 assert fixed(q)==body
 return dict(Actor_Adam=len(trace),new_RL=0,new_Critic=0,before=before,after=metrics(q,data),trace=trace,duals=q.keep_duals.tolist(),frozen_body_sha=body)

def offline_gate(q,data,p):
 m=metrics(q,data);initial=metrics(fresh(p),data)
 return dict(passed=all(x['command_MAE']<=p['teacher_relative_gate']*b['command_MAE'] for x,b in zip(m['teacher'],initial['teacher'])) and max(m['keep_max_by_history'])<=p['keep_max_tolerance'],metrics=m,initial=initial,finite_history_only=True,not_safety_certificate=True)

def export(obj,p,path):
 q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=Prior.ARCH,base_model=p['base_model'],continual=q.continual.state_dict(),anchor_hash=digest(q.anchor.state_dict()),actor_opt=s['actor_opt'],keep_duals=s['keep_duals'],policy_hash=q.phash(),joint_hash=digest(s)),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
