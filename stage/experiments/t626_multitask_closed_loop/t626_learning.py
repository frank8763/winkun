"""Task-balanced critic-free clipped PG, real fresh episodes, fixed own memory."""
from pathlib import Path
import os,sys,copy,time
REQUESTED_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES','')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t620_recurrent_prior_td3'))
import t620_learning as Base
os.environ['CUDA_VISIBLE_DEVICES']=REQUESTED_CUDA
L=Base.Old;torch,np=Base.torch,Base.np
read,save,sha,digest=Base.read,Base.save,Base.sha,Base.digest
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t626_multitask_closed_loop';DOC=ROOT/'stage/docs/day_20261117/t626'
def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);return c
def configure_gpu(p):
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 real=sys.modules['t493_runtime']._real_affinity;os.sched_setaffinity=real;real(0,set(p['stage_cpus']));assert sorted(os.sched_getaffinity(0))==p['stage_cpus'] and torch.cuda.is_available()
def make(p,device='cuda'):
 q=L.frozen(p,p['student_initial']['path']);assert q.phash()==p['student_initial']['policy_hash'];q.requires_grad_(False);q.branch.requires_grad_(True);q.to(device);q.branch.train();assert q.branch.core.dropout==0
 return q,torch.optim.Adam(L.variables(q),lr=p['actor_lr'],foreach=False,fused=False)
def state(obj):return L.state(obj)
def load(obj,s):L.load(obj,s)
def batch(q,groups,p,device='cuda'):
 items=[];diagnostics=[]
 for group in groups:
  assert len(group)==4;parts=[]
  for a in group:
   assert a['stochastic'].all() and not a['ownership'].any() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
   x=torch.tensor(a['states'],device=device);raw=torch.tensor(a['raw_actions'],device=device);atom=torch.tensor(a['atom'],device=device);old=torch.tensor(a['logp'],device=device)
   with torch.no_grad():
    loc=q.sequence(x);err=float((loc-torch.tensor(a['locations'],device=device)).abs().max());de=float((L.P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max());assert err<1e-10 and de<1e-9,(err,de)
   dt=a['native_counts'].astype(np.int64);t=np.r_[0,np.cumsum(dt)[:-1]];ret=np.empty(len(dt));v=0.
   for k in range(len(dt)-1,-1,-1):v=float(a['rewards'][k])+p['gamma_native']**int(dt[k])*v;ret[k]=v
   parts.append(dict(x=x,raw=raw,atom=atom,old=old,initial=loc.detach(),time=t,end=int(dt.sum()),rtg=ret))
  for i,b in enumerate(parts):
   peers=[]
   for j,o in enumerate(parts):
    if i==j:continue
    idx=np.searchsorted(o['time'],b['time'],side='right')-1;valid=b['time']<o['end'];v=np.zeros(len(b['time']));v[valid]=o['rtg'][idx[valid]];peers.append(v)
   b['adv0']=p['gamma_native']**b['time']*(b['rtg']-np.mean(peers,axis=0))
  scale=np.sqrt(sum(float(np.mean(b['adv0']**2)) for b in parts)/4)+1e-8
  for b in parts:b['adv']=b['old'].new_tensor(b['adv0']/scale)
  diagnostics.append(dict(returns=[float(b['rtg'][0]) for b in parts],peer_RMS=float(scale),initial_density_reconstructed=True));items.append(parts)
 return items,diagnostics
def memories(q,p,device='cuda'):
 rows=read(p['memory_manifest'])['memories'];selected=[]
 with torch.no_grad():
  for r in rows:
   if r['kind']!='old' or r['job_id'] not in p['before_success_ids']:continue
   assert sha(r['path'])==r['sha256'];a=Base.G.E.N.S.arrays(r['path']);x=torch.tensor(a['states'],device=device);mask=torch.tensor(~a['ownership'].astype(bool),device=device);target=q.sequence(x).detach();selected.append(dict(id=r['job_id'],x=x,mask=mask,target=target))
 assert len(selected)==33
 return selected
def objective(q,items,refs,p):
 loss=next(q.branch.parameters()).new_zeros(());pg=0.;keep=0.
 for group in items:
  for b in group:
   loc=q.sequence(b['x']);z=L.P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old'];ratio=z.exp();value=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/(len(items)*4);loss=loss+value;pg+=float(value.detach())
 for r in refs:
  loc=q.sequence(r['x'])[r['mask']];target=r['target'][r['mask']];value=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().mean()/len(refs);loss=loss+p['keep_weight']*value;keep+=float(value.detach())
 return loss,dict(policy_loss=pg,keep=keep)
def trust(q,items,p):
 result=[]
 with torch.no_grad():
  for group in items:
   values=[]
   for b in group:
    per_step=.5*((q.sequence(b['x'])-b['initial'])/b['initial'].new_tensor(p['raw_sigma'])).square().sum(-1);values.append(float(per_step.mean()))
   result.append(sum(values)/4)
 return result
def fit(obj,groups,p,guard,event,steps=None):
 q,opt=obj;body=L.fixed(q);items,diag=batch(q,groups,p);refs=memories(q,p);trace=[];applied=0
 for k in range(p['actor_steps'] if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,detail=objective(q,items,refs,p);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);kl=trust(q,items,p);okay=max(kl)<=p['max_task_latent_KL'] and np.isfinite(kl).all()
  if not okay:load(obj,prior)
  else:applied+=1
  trace.append(dict(step=k,loss=float(loss.detach()),**detail,gradient_norm=norm,task_latent_KL=kl,accepted=bool(okay)))
  if not okay:break
 assert L.fixed(q)==body
 return dict(attempted_Adam=len(trace),applied=applied,trace=trace,peer_batch=diag,critic_updates=0,reference_success_histories=33,missing_known_success_history='wide117_8m_dev6_LEFT_17_is_covered_by_new_collection_and_final_closed_loop',task_balanced=True,full_current_history_BPTT=True,not_standard_value_GAE_PPO=True,not_claimed_unbiased_variance_reduction=True)
