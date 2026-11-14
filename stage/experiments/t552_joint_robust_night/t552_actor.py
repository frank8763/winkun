"""Fixed-size trainable smooth readout on the immutable T550 causal Actor.

Delta checkpoints contain only this readout and its optimizer, never repeated
copies of the frozen backbone or a deployment teacher/task router.
"""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'));import t550_actor as Old
P,G,torch,np=Old.P,Old.G,Old.torch,Old.np
read,save,sha,digest=Old.read,Old.save,Old.sha,Old.digest
ARCH='T552_single120_64carry_immutable_T550_smooth5986_readout'
class Actor(Old.Actor):
 def __init__(self,p,old_n,added_n,sizes):
  super().__init__(p,old_n,added_n,sizes)
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(p['readout_seed']);self.extra=torch.nn.Sequential(torch.nn.Linear(184,32),torch.nn.Tanh(),torch.nn.Linear(32,2)).double();torch.nn.init.zeros_(self.extra[-1].weight);torch.nn.init.zeros_(self.extra[-1].bias)
  self.requires_grad_(False)
 def location(self,h,n):return super().location(h,n)+self.p['readout_amplitude']*torch.tanh(self.extra(torch.cat([h,n],-1)))
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma'],amplitude=self.p['readout_amplitude']))
def initial(p):
 q0=Old.frozen(p,p['frozen_base']['path']);assert q0.phash()==p['frozen_base']['policy_hash'];q=Actor(p,len(q0.repair_centers),len(q0.added_centers),[len(l.centers) for l in q0.layers]);missing,extra=q.load_state_dict(q0.state_dict(),strict=False);assert set(missing)=={'extra.'+k for k in q.extra.state_dict()} and not extra;return q

def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Old.frozen(p,path)
 assert s['frozen_base']['sha256']==p['frozen_base']['sha256'] and sha(s['frozen_base']['path'])==s['frozen_base']['sha256'];q=initial(p);q.extra.load_state_dict(s['extra']);assert q.phash()==s['policy_hash'];return q.eval()

def make(p):
 q=frozen(p,p['initial']['path']);assert isinstance(q,Actor);q.requires_grad_(False);q.extra.requires_grad_(True);opt=torch.optim.Adam(q.extra.parameters(),lr=p['bc_lr'],foreach=False,fused=False);s=P.checkpoint(p['initial']['path']);opt.load_state_dict(s['actor_opt']);assert sum(x.numel() for x in q.extra.parameters())==5986;return q,opt

def state(obj):return dict(extra=copy.deepcopy(obj[0].extra.state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].extra.load_state_dict(s['extra']);obj[1].load_state_dict(s['actor_opt'])
def fixed_hash(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('extra.')})
def export(obj,p,path):
 q,opt=obj;s=state(obj);buf=io.BytesIO();torch.save(dict(architecture=ARCH,frozen_base=p['frozen_base'],extra=s['extra'],actor_opt=s['actor_opt'],joint_hash=digest(s),policy_hash=q.phash(),single_actor=True,no_runtime_teacher=True),buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
def zero_export(p,path):
 q=initial(p);opt=torch.optim.Adam(q.extra.parameters(),lr=p['bc_lr'],foreach=False,fused=False);return export((q,opt),p,path)

def feature_mean(q,x):
 with torch.no_grad():
  n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);f=torch.cat([h[:,0],n],-1);base=super(Actor,q).location(h[:,0],n).detach();return f,base

def cached_dataset(p,limit=None):
 base=Old.frozen(p,p['frozen_base']['path']);data=[];pairs=[]
 with torch.no_grad():
  for kind,rows in [('restore',p['restore_rows']),('keep',p['keep_rows']),('neighbor',p['neighbor_rows'])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);n=base.normalized(x);h,_=base.core(base.encoder(n)[:,None]);f=torch.cat([h[:,0],n],-1);loc=base.sequence(x);target=torch.tensor(a['actions'] if kind=='restore' else a['locations'],dtype=torch.float64);mask=torch.tensor(~a['ownership'].astype(bool)) if kind=='neighbor' else torch.ones(len(x),dtype=torch.bool)
    if kind!='restore':assert float((loc-target).abs().max())<1e-10,(kind,row['job_id'])
    if limit is not None:mask[limit:]=False
    assert mask.any();data.append(dict(id=row['job_id'],kind=kind,f=f[mask],base=loc[mask].detach(),target=target[mask]))
  for pair in p['pair_rows']:
   aa=G.E.N.S.arrays(pair['baseline']['path']);bb=G.E.N.S.arrays(pair['neighbor']['path']);k=int(bb['intervention_hold'])+1;end=min(k+20,len(aa['states']),len(bb['states']));assert end>k
   def ff(a):
    x=torch.tensor(a['states']);n=base.normalized(x);h,_=base.core(base.encoder(n)[:,None]);return torch.cat([h[:,0],n],-1)
   f,g=ff(aa)[k:end],ff(bb)[k:end];d=(f-g).square().sum(-1);mask=d>1e-8;assert mask.any();pairs.append(dict(f=f[mask],g=g[mask],den=d[mask]+1e-6))
 assert len(data)==33 and len(pairs)==4;return data,pairs

def location(q,g):return g['base']+q.p['readout_amplitude']*torch.tanh(q.extra(g['f']))
def supervised(q,data,pairs,p):
 counts={kind:sum(g['kind']==kind for g in data) for kind in ['restore','keep','neighbor']};total=0.;values={kind:0. for kind in counts}
 for g in data:
  loc=location(q,g)
  if g['kind']=='restore':err=P.action_loss(loc,g['target']).mean(-1)
  else:err=(loc-g['target']).square().mean(-1)
  loss=(.5*err.mean()+.5*err.topk(max(1,__import__('math').ceil(.2*len(err)))).values.mean())/counts[g['kind']];total=total+loss;values[g['kind']]+=float(loss.detach())
 local=0.
 if p['arm']=='response':
  for pair in pairs:
   a=p['readout_amplitude']*torch.tanh(q.extra(pair['f']));b=p['readout_amplitude']*torch.tanh(q.extra(pair['g']));v=((a-b).square().sum(-1)/pair['den']).mean()/len(pairs);total=total+p['pair_weight']*v;local+=float(v.detach())
 values['local_response']=local;return total,values

def bc(obj,data,pairs,p,guard,event,steps=None):
 q,opt=obj;before=fixed_hash(q);trace=[]
 for group in opt.param_groups:group['lr']=p['bc_lr']
 for k in range(p['bc_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);loss,v=supervised(q,data,pairs,p);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.extra.parameters(),1.,error_if_nonfinite=True));event('BC_BEFORE',k);opt.step();event('BC_AFTER',k);trace.append(dict(step=k,loss=float(loss.detach()),norm=norm,**v))
 assert fixed_hash(q)==before;return trace

def rl_or_control(obj,arrays,data,pairs,p,guard,event):
 sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
 q,opt=obj;items,diag=Credit.peer_batch(obj,arrays,p);cached=[]
 for b in items:
  f,base=feature_mean(q,b['x']);cached.append(dict(b, f=f,base=base))
 trace=[];fixed=fixed_hash(q)
 for group in opt.param_groups:group['lr']=p['rl_lr']
 for k in range(p['rl_steps']):
  guard();opt.zero_grad(set_to_none=True);loss,v=supervised(q,data,pairs,p);pl=0.
  if p['arm']!='supervised':
   for b in cached:
    loc=location(q,b);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();term=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(cached);assert torch.isfinite(term);loss=loss+term;pl+=float(term.detach())
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.extra.parameters(),1.,error_if_nonfinite=True));event('RL_BEFORE' if p['arm']!='supervised' else 'CONTROL_BEFORE',k);opt.step();event('RL_AFTER' if p['arm']!='supervised' else 'CONTROL_AFTER',k)
  with torch.no_grad():
   z=[P.L.density(location(q,b),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in cached];kl=sum(float((x.exp()-1-x).mean()) for x in z)/len(z)
  trace.append(dict(step=k,total_loss=float(loss.detach()),policy_loss=pl,norm=norm,KL=kl,**v))
  if kl>p['target_kl']:break
 assert fixed_hash(q)==fixed;return dict(trace=trace,peer_batch=diag,RL_Adam=0 if p['arm']=='supervised' else len(trace),BC_control_Adam=len(trace) if p['arm']=='supervised' else 0,new_Value=0)
