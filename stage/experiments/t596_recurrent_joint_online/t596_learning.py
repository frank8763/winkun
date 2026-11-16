"""Exact T550 initial function; train encoder/GRU/neural command layers.

Critic-free clipped PG with the existing physical-time peer baseline. This is
PPO-style, not a claimed standard value/GAE PPO implementation or CLEAR.
"""
from pathlib import Path
import sys,copy,gzip,io,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t550_offline_obstacle_relay'));import t550_actor as Parent
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
P,G,torch,np=Parent.P,Parent.G,Parent.torch,Parent.np
read,save,sha,digest=Parent.read,Parent.save,Parent.sha,Parent.digest
PREFIXES=('encoder.','core.','head.','drive_adapter.','steer_adapter.')
ARCH='T596_exact_T550_initial_trainable_recurrent_neural_layers'
def fresh(p):
 assert sha(p['base_model']['path'])==p['base_model']['sha256'];q=Parent.frozen(p,p['base_model']['path']);assert q.phash()==p['base_model']['policy_hash'];return q
def variables(q):return [v for k,v in q.named_parameters() if k.startswith(PREFIXES)]
def fixed(q):return digest({k:v for k,v in q.state_dict().items() if not k.startswith(PREFIXES)})
def make(p):
 q=fresh(p);q.requires_grad_(False)
 for v in variables(q):v.requires_grad_(True)
 return q,torch.optim.Adam(variables(q),lr=p['rl_lr'],foreach=False,fused=False)
def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])
def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Parent.frozen(p,path)
 assert s['base_model']==p['base_model'];q=fresh(p);expected={k for k in q.state_dict() if k.startswith(PREFIXES)};assert set(s['trainable_state'])==expected
 q.load_state_dict(s['trainable_state'],strict=False);assert q.phash()==s['policy_hash'] and fixed(q)==s['fixed_sha'];return q.eval()
def references(p):
 q=fresh(p);data=[]
 with torch.no_grad():
  for kind,rows in [('old',p['keep_rows']),('recovery',p['recovery_rows'])]:
   for row in rows:
    assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);own=a['ownership'].astype(bool);mask=torch.tensor(~own)
    if kind=='recovery':assert 'student_takeover_first' in a and own[:int(a['student_takeover_first'])].all() and not own[int(a['student_takeover_first']):].any()
    assert mask.any();target=q.sequence(x).clone() if kind=='old' else torch.tensor(a['actions'],dtype=torch.float64)
    data.append(dict(id=row['job_id'],kind=kind,x=x,target=target,mask=mask,sha256=row['sha256']))
 assert len(data)==42 and sum(g['kind']=='old' for g in data)==39
 return data
def losses(q,data,p,backward):
 out=dict(old=0.,recovery=0.)
 for g in data:
  z=q.sequence(g['x'])[g['mask']];t=g['target'][g['mask']]
  if g['kind']=='old':per=.5*((z-t)/z.new_tensor(p['raw_sigma'])).square().mean(-1);weight=p['keep_weight']/39
  else:per=P.action_loss(z,t).mean(-1);weight=p['recovery_weight']/3
  loss=(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())*weight
  if backward:loss.backward()
  out[g['kind']]+=float(loss.detach())
 return out
def metrics(q,data):
 rows=[]
 with torch.no_grad():
  for g in data:
   cmd=q.sequence(g['x'])[g['mask']].clamp(-1,1).float().double();target=g['target'][g['mask']].clamp(-1,1).float().double();e=cmd-target
   rows.append(dict(id=g['id'],kind=g['kind'],RMS=float(e.square().mean().sqrt()),max=float(e.abs().max())))
 return dict(rows=rows,old_worst_RMS=max(r['RMS'] for r in rows if r['kind']=='old'),recovery_worst_RMS=max(r['RMS'] for r in rows if r['kind']=='recovery'))
def fit(obj,arrays,data,p,guard,event,steps=None):
 q,opt=obj;body=fixed(q);assert len(arrays)==8;groups=[];diagnostics=[]
 for i in range(2):
  items,diag=Credit.peer_batch(obj,arrays[4*i:4*i+4],p);groups.append(items);diagnostics.append(diag)
 before=metrics(q,data);trace=[]
 for k in range(p['rl_steps'] if steps is None else steps):
  guard();opt.zero_grad(set_to_none=True);pl=0.
  for items in groups:
   for b in items:
    loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/8;loss.backward();pl+=float(loss.detach())
  aux=losses(q,data,p,True);norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
  with torch.no_grad():
   kls=[sum(float(((z:=(P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in items)/4 for items in groups]
  trace.append(dict(step=k,policy_loss=pl,aux=aux,gradient_norm=norm,task_KLs=kls))
  if max(kls)>p['target_kl']:break
 assert fixed(q)==body
 return dict(Actor_Adam=len(trace),shadow_separately_counted=True,Value_Adam=0,trace=trace,before=before,after=metrics(q,data),peer_batches=diagnostics,trainable_parameters=sum(v.numel() for v in variables(q)),full_current_history_BPTT=True,old_and_new_fresh_policy_data=True,not_standard_value_GAE_PPO=True,not_claimed_unbiased_or_variance_reduced=True,fixed_sha=body)
def export(obj,p,path):
 q,opt=obj;s=state(obj);b=io.BytesIO();torch.save(dict(architecture=ARCH,base_model=p['base_model'],trainable_state={k:copy.deepcopy(v) for k,v in q.state_dict().items() if k.startswith(PREFIXES)},actor_opt=s['actor_opt'],policy_hash=q.phash(),joint_hash=digest(s),fixed_sha=fixed(q),single_actor=True,no_runtime_teacher=True),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(s))
