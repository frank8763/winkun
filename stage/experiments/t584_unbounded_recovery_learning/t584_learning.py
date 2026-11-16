"""Finite supervised recovery, same initial function, remove artificial .1 range cap."""
from pathlib import Path
import sys,copy,gzip,io
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t581_preserved_t550_rl'));import t581_learning as Old
P,G,torch,np=Old.P,Old.G,Old.torch,Old.np
read,save,sha,digest=Old.read,Old.save,Old.sha,Old.digest
state,load,fixed,feature=Old.state,Old.load,Old.fixed,Old.feature
ARCH='T584_EXACT_T550_PLUS_ZERO_INITIAL_UNBOUNDED_SMOOTH_RESIDUAL'
class Actor(Old.Actor):
 def location(self,h,n):return Old.Parent.Actor.location(self,h,n)+self.continual(torch.cat([h,n],-1))
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))
def fresh(p):
 parent=Old.ORIGINAL_FROZEN(p,p['base_model']['path']);return Actor(p,parent)
def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Old.frozen(p,path)
 assert s['base_model']==p['base_model'];q=fresh(p);q.continual.load_state_dict(s['continual']);assert q.phash()==s['policy_hash'];return q.eval()
def make(p):
 q=fresh(p);q.continual.requires_grad_(True);return q,torch.optim.Adam(q.continual.parameters(),lr=p['BC_lr'],foreach=False,fused=False)
def loc(q,f,b):return b+q.continual(f)
def dataset(p):
 q=fresh(p);keep=[];teach=[]
 for r in p['keep_rows']:
  a=G.E.N.S.arrays(r['path']);f,b=feature(q,torch.tensor(a['states']));keep.append(dict(f=f,b=b,target=b.clone()))
 for r in p['recovery_rows']:
  a=G.E.N.S.arrays(r['path']);k=int(a['teacher_first_declared']);assert r['success'] and not r['collision'] and a['ownership'][k:].all()
  f,b=feature(q,torch.tensor(a['states']));teach.append(dict(f=f[k:],b=b[k:],target=torch.tensor(a['actions'][k:],dtype=torch.float64)))
 assert len(keep)==27 and len(teach)==3;return keep,teach

def metrics(q,data):
 keep,teach=data
 with torch.no_grad():
  return dict(teacher=[dict(command_MAE=float((loc(q,g['f'],g['b']).clamp(-1,1)-g['target']).abs().mean()),command_max=float((loc(q,g['f'],g['b']).clamp(-1,1)-g['target']).abs().max())) for g in teach],
   keep_location_MAE=float(torch.cat([(loc(q,g['f'],g['b'])-g['target']).abs().flatten() for g in keep]).mean()))
def fit(obj,data,p,guard,event,steps=None):
 q,opt=obj;body=fixed(q);keep,teach=data;before=metrics(q,data);trace=[]
 def merge(groups):
  return dict(f=torch.cat([g['f'] for g in groups]),b=torch.cat([g['b'] for g in groups]),target=torch.cat([g['target'] for g in groups]),weights=torch.cat([torch.full((len(g['f']),),1/(len(groups)*len(g['f'])),dtype=torch.float64) for g in groups]))
 km,tm=merge(keep),merge(teach)
 for k in range(p['BC_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);tl=kl=0.
  for g,weight,is_keep in [(tm,1.,False),(km,p['keep_weight'],True)]:
   v=(.5*((loc(q,g['f'],g['b'])-g['target'])/.2).square().mean(-1)*g['weights']).sum();(weight*v).backward()
   if is_keep:kl=float(v.detach())
   else:tl=float(v.detach())
  norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([tl,kl,norm])
 assert fixed(q)==body
 return dict(Actor_Adam=len(trace),new_RL=0,new_Critic=0,before=before,after=metrics(q,data),trace=trace,frozen_body_sha=body)
def export(obj,p,path):
 q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,base_model=p['base_model'],continual=q.continual.state_dict(),actor_opt=s['actor_opt'],policy_hash=q.phash(),joint_hash=digest(s)),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
