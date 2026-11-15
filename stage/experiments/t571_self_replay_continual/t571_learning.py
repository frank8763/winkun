"""Task-balanced own-state rehearsal plus verified recovery imitation.

Frozen full-history encoder/GRU and original smooth policy; one new smooth
residual, no compact bank, task ID, runtime teacher or planning route.
"""
from pathlib import Path
import sys,copy,gzip,io
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t562_full_history_representation'));import t562_actor as Base
P,G,torch,np=Base.P,Base.G,Base.torch,Base.np
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest
ARCH='T571_single120_real64carry_frozenT562_5986smooth_continual'
class Actor(Base.Actor):
 def __init__(self,p):
  super().__init__(p)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(571);self.continual=torch.nn.Sequential(torch.nn.Linear(184,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).double();torch.nn.init.zeros_(self.continual[-1].weight);torch.nn.init.zeros_(self.continual[-1].bias)
  self.requires_grad_(False)
 def location(self,h,n):return super().location(h,n)+self.continual(torch.cat([h,n],-1))
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))
def make(p):
 q=Actor(p);saved=P.checkpoint(p['initial']['path']);assert sha(p['initial']['path'])==p['initial']['sha256'];missing,unexpected=q.load_state_dict(saved['actor'],strict=False);assert set(missing)=={'continual.'+k for k in q.continual.state_dict()} and not unexpected
 q.continual.requires_grad_(True);assert sum(v.numel() for v in q.parameters() if v.requires_grad)==5986;return q,torch.optim.Adam(q.continual.parameters(),lr=p['fit_lr'],foreach=False,fused=False)
def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Base.frozen(p,path)
 q=Actor(p);q.load_state_dict(s['actor']);assert q.phash()==s['policy_hash'];return q.eval()
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.')})
def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])
def feature(q,x):
 with torch.no_grad():
  n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);f=torch.cat([h[:,0],n],-1);base=Base.Actor.location(q,h[:,0],n)
 return f,base
def dataset(p,memory):
 q=make(p)[0];data=[]
 for row in memory:
  assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);f,b=feature(q,x);target=torch.tensor(a['locations']);assert float((q.sequence(x)-target).abs().max())<=1e-9
  data.append(dict(id=row['job_id'],kind='keep',f=f,base=b,target=target))
 for row in p['repair_rows']:
  assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);f,b=feature(q,x);mask=a['ownership'].astype(bool) if row['teacher_suffix_only'] else np.ones(len(x),bool);assert mask.any();target=torch.tensor(a['teacher_actions'] if row['teacher_suffix_only'] else a['actions'],dtype=torch.float64)
  data.append(dict(id=row['job_id'],kind='repair',f=f[mask],base=b[mask],target=target[mask]))
 assert len(data)==23 and sum(g['kind']=='keep' for g in data)==21
 return data
def loc(q,g):return g['base']+q.continual(g['f'])
def measure(q,data):
 with torch.no_grad():
  return [dict(id=g['id'],kind=g['kind'],location_MSE=float((loc(q,g)-g['target']).square().mean()),command_RMS=float((loc(q,g).clamp(-1,1).float().double()-g['target'].clamp(-1,1)).square().mean().sqrt())) for g in data]
def fit(obj,data,p,guard,event,steps=None):
 q,opt=obj;body=fixed(q);trace=[];before=measure(q,data);keep=[g for g in data if g['kind']=='keep'];repair=[g for g in data if g['kind']=='repair'];assert len(keep)==21 and len(repair)==2
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);loss=q.head.weight.new_zeros(())
  for g in keep:loss=loss+(loc(q,g)-g['target']).square().mean()/len(keep)
  for g in repair:
   err=P.action_loss(loc(q,g),g['target']).mean(-1);loss=loss+(.5*err.mean()+.5*err.topk(max(1,int(.2*len(err)))).values.mean())/len(repair)
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);trace.append([float(loss.detach()),norm]);assert fixed(q)==body
 return dict(steps=len(trace),trace=trace,before=before,after=measure(q,data),frozen_policy_body_hash=body,new_RL=0,full_history_features_frozen_body=True)
def export(obj,p,path):
 s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,actor=s['actor'],actor_opt=s['actor_opt'],policy_hash=obj[0].phash(),joint_hash=digest(s),single_actor=True,teacher_runtime=False),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as h:h.write(blob);h.flush();__import__('os').fsync(h.fileno())
 return dict(path=str(path),sha256=sha(path),policy_hash=obj[0].phash(),joint_hash=digest(s),bytes=len(blob))
