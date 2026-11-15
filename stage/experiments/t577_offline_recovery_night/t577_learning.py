"""Compact neural-head checkpoints; joint actual-prefix recovery and old-history learning."""
from pathlib import Path
import sys,io,gzip,copy,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t576_nonlinear_plasticity_pair'));import t576_learning as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
ARCH='T577_COMPACT_EXISTING_T571_HEAD_ON_FROZEN_T576_BACKBONE'

def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Prior.frozen(p,path)
 assert s['static_model']==p['static_model'] and sha(s['static_model']['path'])==s['static_model']['sha256'];q=Prior.frozen(p,s['static_model']['path']);q.continual.load_state_dict(s['continual']);assert q.phash()==s['policy_hash'];return q.eval()
def make(p):
 q=frozen(p,p['initial']['path']);q.requires_grad_(False);q.continual.requires_grad_(True);opt=torch.optim.Adam(q.continual.parameters(),lr=p['bc_lr'],foreach=False,fused=False)
 s=P.checkpoint(p['initial']['path'])
 if s.get('architecture')==ARCH:opt.load_state_dict(s['actor_opt'])
 return q,opt
state,load=Prior.state,Prior.load

def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith('continual.')})
def export(obj,p,path):
 q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,static_model=p['static_model'],continual=q.continual.state_dict(),actor_opt=s['actor_opt'],policy_hash=q.phash(),joint_hash=digest(s),single_actor=True,teacher_at_deployment=False),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))

def dataset(p):
 # Frozen body features from full causal histories; anchors refer to exact
 # T572A locations on its own states, not fabricated current-policy targets.
 q=frozen(p,p['initial']['path']);anchor=frozen(p,p['anchor_model']['path']);data=[]
 with torch.no_grad():
  for kind,rows in [('keep',p['memory_rows']),('repair',p['correction_rows'])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);f,b=Prior.Prior.feature(q,torch.tensor(a['states']));initial=b+q.continual(f)
    if kind=='keep':
     target=torch.tensor(a['locations']);assert float((anchor.sequence(torch.tensor(a['states']))-target).abs().max())<1e-9;mask=torch.ones(len(f),dtype=torch.bool);command=target.clamp(-1,1).float().double()
    else:
     assert row['success'] and not row['collision'];mask=torch.tensor(a['ownership'].astype(bool));assert mask.any();command=torch.tensor(a['teacher_actions'],dtype=torch.float64);target=torch.where(command>=1,torch.maximum(initial,torch.ones_like(initial)),torch.where(command<=-1,torch.minimum(initial,-torch.ones_like(initial)),command))
    data.append(dict(id=row['job_id'],kind=kind,f=f[mask],base=b[mask],target=target[mask],command=command[mask]))
 assert len(p['memory_rows'])==22 and data;return data

def supervised(q,data,p):
 losses={'keep':[],'repair':[]};diag=[]
 for g in data:
  e=(g['base']+q.continual(g['f'])-g['target']).square().mean(-1)
  v=e.mean() if p['arm']=='mean' else .5*e.mean()+.5*e.topk(max(1,math.ceil(.2*len(e)))).values.mean()
  losses[g['kind']].append(v);diag.append(float(v.detach()))
 total=q.head.weight.new_zeros(())
 for kind,values in losses.items():
  if values:
   z=torch.stack(values);total+=z.mean() if p['arm']=='mean' else .5*z.mean()+.5*z.max()
 return total,diag

def bc(obj,data,p,guard,event,steps=None):
 q,opt=obj;body=fixed(q);trace=[]
 for g in opt.param_groups:g['lr']=p['bc_lr']
 for k in range(p['bc_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);loss,diag=supervised(q,data,p);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True));event('BC_BEFORE',k);opt.step();event('BC_AFTER',k);trace.append([float(loss.detach()),norm,max(diag)])
 assert fixed(q)==body;return trace

def rl(obj,arrays,data,p,guard,event):
 sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
 q,opt=obj;items,diag=Credit.peer_batch(obj,arrays,p);cache=[]
 with torch.no_grad():
  for b in items:
   f,base=Prior.Prior.feature(q,b['x']);cache.append(dict(b,f=f,base=base))
 for g in opt.param_groups:g['lr']=p['rl_lr']
 body=fixed(q);trace=[]
 for k in range(p['rl_steps']):
  guard();opt.zero_grad(set_to_none=True);loss,_=supervised(q,data,p);pl=0.
  for b in cache:
   loc=b['base']+q.continual(b['f']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();v=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(cache);loss+=v;pl+=float(v.detach())
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.continual.parameters(),1.,error_if_nonfinite=True));event('RL_BEFORE',k);opt.step();event('RL_AFTER',k)
  with torch.no_grad():
   zz=[P.L.density(b['base']+q.continual(b['f']),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in cache];kl=sum(float((z.exp()-1-z).mean()) for z in zz)/len(zz)
  trace.append(dict(loss=float(loss.detach()),policy_loss=pl,norm=norm,KL=kl))
  if kl>p['target_kl']:break
 assert fixed(q)==body;return dict(trace=trace,peer_batch=diag,actual_onpolicy_Adam=len(trace),not_unbiased_or_universal_variance_claim=True)
