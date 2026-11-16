"""Function-preserving subtraction of a frozen initial nonlinear residual."""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t584_unbounded_recovery_learning'));import t584_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
feature,dataset=Prior.feature,Prior.dataset
ARCH='T586_EXACT_T550_PLUS_TRAINABLE_MINUS_FROZEN_INITIAL_MLP'
class Actor(Prior.Actor):
 def __init__(self,p,parent):
  super().__init__(p,parent)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(p['residual_seed']);self.anchor=torch.nn.Sequential(torch.nn.Linear(184,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).double()
  self.continual.load_state_dict(self.anchor.state_dict());self.requires_grad_(False)
 def location(self,h,n):
  z=torch.cat([h,n],-1);return Prior.Old.Parent.Actor.location(self,h,n)+(self.continual(z)-self.anchor(z))
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))
def fresh(p):return Actor(p,Prior.Old.ORIGINAL_FROZEN(p,p['base_model']['path']))
def loc(q,f,b):return b+(q.continual(f)-q.anchor(f))
def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Prior.frozen(p,path)
 assert s['base_model']==p['base_model'];q=fresh(p);q.continual.load_state_dict(s['continual']);assert digest(q.anchor.state_dict())==s['anchor_hash'];assert q.phash()==s['policy_hash'];return q.eval()
def make(p):
 q=fresh(p);q.continual.requires_grad_(True);assert sum(x.numel() for x in q.parameters() if x.requires_grad)==5986
 return q,torch.optim.Adam(q.continual.parameters(),lr=p['BC_lr'],foreach=False,fused=False)
def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=None if obj[1] is None else copy.deepcopy(obj[1].state_dict()))
def load(obj,s):
 obj[0].load_state_dict(s['actor'])
 if obj[1] is not None:obj[1].load_state_dict(s['actor_opt'])
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.')})
def metrics(q,data):
 keep,teach=data
 with torch.no_grad():return dict(teacher=[dict(command_MAE=float((loc(q,g['f'],g['b']).clamp(-1,1)-g['target']).abs().mean()),command_max=float((loc(q,g['f'],g['b']).clamp(-1,1)-g['target']).abs().max())) for g in teach],keep_location_MAE=float(torch.cat([(loc(q,g['f'],g['b'])-g['target']).abs().flatten() for g in keep]).mean()))
def fit(obj,data,p,guard,event,steps=None):
 q,opt=obj;body=fixed(q);keep,teach=data;before=metrics(q,data);trace=[]
 def merge(groups):return dict(f=torch.cat([g['f'] for g in groups]),b=torch.cat([g['b'] for g in groups]),target=torch.cat([g['target'] for g in groups]),weights=torch.cat([torch.full((len(g['f']),),1/(len(groups)*len(g['f'])),dtype=torch.float64) for g in groups]))
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
 q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,base_model=p['base_model'],continual=q.continual.state_dict(),anchor_hash=digest(q.anchor.state_dict()),actor_opt=s['actor_opt'],policy_hash=q.phash(),joint_hash=digest(s)),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
def offline_gate(q,data,p):
 a=Prior.frozen(p,p['historical_A_model']['path']);m=metrics(q,data);old=Prior.metrics(a,data)
 assert old==p['historical_A_fit']
 with torch.no_grad():
  aa=[float((Prior.loc(a,g['f'],g['b'])-g['target']).abs().max()) for g in data[0]];bb=[float((loc(q,g['f'],g['b'])-g['target']).abs().max()) for g in data[0]]
 return dict(passed=all(b['command_MAE']<a['command_MAE'] for a,b in zip(old['teacher'],m['teacher'])) and all(b<=a+1e-12 for a,b in zip(aa,bb)),B=m,A=old,A_keep_max_by_history=aa,B_keep_max_by_history=bb,historical_A_NOT_new_independent_replication=True,tolerance=1e-12)
