"""Fresh-policy two-step PG; sigma-aware identity and fixed T608 memory target."""
from pathlib import Path
import sys,os,copy,io,gzip,time
REQUESTED_CUDA=os.environ.get('CUDA_VISIBLE_DEVICES','')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t626_multitask_closed_loop'))
import t626_learning as P
os.environ['CUDA_VISIBLE_DEVICES']=REQUESTED_CUDA
L,torch,np=P.L,P.torch,P.np
read,save,sha,digest=P.read,P.save,P.sha,P.digest
HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t628r_memory_start_recovery';DOC=ROOT/'stage/docs/day_20261117/t628r'
ARCH='T628R_T608_branch_sigma_explicit'
ORIGINAL_FROZEN=L.frozen
def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);return c
def frozen(p,path):
 s=L.P.checkpoint(path);canonical=dict(p,raw_sigma=[.2,.2])
 if s.get('architecture')!=ARCH:return ORIGINAL_FROZEN(canonical,path)
 parent=s['parent'];assert sha(parent['path'])==parent['sha256'];q=ORIGINAL_FROZEN(canonical,parent['path']);q.branch.load_state_dict(s['branch']);q.p=copy.deepcopy(p);q.p['raw_sigma']=s['sigma'];assert q.phash()==s['policy_hash'] and L.fixed(q)==s['fixed_sha'];return q.eval().requires_grad_(False)
def initial(p,sigma):
 q=frozen(p,p['student_initial']['path']);q.p=copy.deepcopy(p);q.p['raw_sigma']=list(sigma);return q
def set_sigma(obj,sigma):obj[0].p['raw_sigma']=list(sigma)
def make(p,sigma,device='cuda'):
 q=initial(p,sigma).to(device);q.requires_grad_(False);q.branch.requires_grad_(True);q.branch.train();return q,torch.optim.SGD(L.variables(q),lr=p['actor_lr'],momentum=0.,foreach=False)
def cpu(value):
 if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
 if isinstance(value,list):return [cpu(v) for v in value]
 if isinstance(value,tuple):return tuple(cpu(v) for v in value)
 return value
def state(obj):return dict(**cpu(L.state(obj)),sigma=list(obj[0].p['raw_sigma']))
def load(obj,s):L.load(obj,s);set_sigma(obj,s['sigma'])
def export(obj,p,path):
 q,opt=obj;s=state(obj);data=dict(architecture=ARCH,parent=p['student_initial'],branch=s['actor_branch'] if 'actor_branch' in s else {k:v for k,v in q.branch.state_dict().items()},actor_opt=s['actor_opt'],sigma=s['sigma'],policy_hash=q.phash(),fixed_sha=L.fixed(q),joint_hash=digest(s));buf=io.BytesIO();torch.save(data,buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob)
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s),sigma=s['sigma'])
def references(q,p):
 ref=initial(p,q.p['raw_sigma']).to(next(q.parameters()).device);rows=read(p['memory_manifest'])['memories'];values=[]
 with torch.no_grad():
  for r in rows:
   if r['kind']!='old' or r['job_id'] not in p['before_success_ids']:continue
   assert sha(r['path'])==r['sha256'];a=P.Base.G.E.N.S.arrays(r['path']);x=torch.tensor(a['states'],device=next(q.parameters()).device);mask=torch.tensor(~a['ownership'].astype(bool),device=x.device);values.append(dict(id=r['job_id'],x=x,mask=mask,target=ref.sequence(x).detach()))
 assert len(values)==33;return values
configure_gpu=P.configure_gpu
def objective(q,items,refs,p):
 loss=next(q.branch.parameters()).new_zeros(());pg=0.;keep=0.
 for group in items:
  for b in group:
   loc=q.sequence(b['x']);ratio=(L.P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();value=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/(len(items)*4);loss=loss+value;pg+=float(value.detach())
 # Keep the physical function regularizer equally scaled in both arms.
 # Exploration sigma still correctly changes sampling, density, and KL trust.
 for r in refs:
  loc=q.sequence(r['x'])[r['mask']];target=r['target'][r['mask']];value=.5*((loc-target)/loc.new_tensor(p['retention_sigma'])).square().mean()/len(refs);loss=loss+p['keep_weight']*value;keep+=float(value.detach())
 return loss,dict(policy_loss=pg,keep=keep)
def fit(obj,groups,p,guard,event):
 q,opt=obj;pc=dict(p,raw_sigma=list(q.p['raw_sigma']));items,diag=P.batch(q,groups,pc);refs=references(q,p);body=L.fixed(q);trace=[];applied=0
 for k in range(p['actor_steps']):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,detail=objective(q,items,refs,pc);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);kl=P.trust(q,items,pc);okay=np.isfinite(kl).all() and max(kl)<=p['max_task_latent_KL']
  if okay:applied+=1
  else:load(obj,prior)
  trace.append(dict(step=k,loss=float(loss.detach()),**detail,gradient_norm=norm,task_latent_KL=kl,accepted=bool(okay)))
  if not okay:break
 assert L.fixed(q)==body
 return dict(attempted_optimizer=len(trace),applied=applied,trace=trace,peer_batch=diag,optimizer='SGD_no_momentum',sigma=list(q.p['raw_sigma']),full_current_history_BPTT=True,reference_is_fixed_T608_not_round_initial=True,no_Critic=True)
