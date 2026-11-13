"""Three preregistered fresh-policy mechanisms, four clipped SGD steps.

A: terminal safe-GOAL leave-one-out score; B: native-time reward-to-go;
C: B restricted to the two neural readouts. Recurrent retention is evaluated
on full histories, never on zero-initialized individual observations. Standard
REINFORCE/PPO/replay ideas; no novelty or safety guarantee is claimed.
"""
from pathlib import Path
import copy,gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t539_terminal_success_pg'));import t539_learning as O
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'));import t519_learning as Credit
A,P,G,torch,np=O.A,O.P,O.G,O.torch,O.np
read,save,sha,digest=O.read,O.save,O.sha,O.digest

def make(p):
 q=A.frozen(p,p['initial']['path']).train();q.requires_grad_(p['arm']!='readout')
 if p['arm']=='readout':q.drive_adapter.requires_grad_(True);q.steer_adapter.requires_grad_(True)
 opt=torch.optim.SGD([x for x in q.parameters() if x.requires_grad],lr=p['learning_rate'],momentum=0.,foreach=False)
 assert q.phash()==p['initial']['policy_hash'] and not opt.state
 return q,opt

def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])

def references(p):
 refs=[]
 for row in p['keep_rows']:
  assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
  refs.append((row['job_id'],torch.tensor(a['states']),torch.tensor(a['locations'])))
 assert len(refs)==19
 return refs

def batch(obj,arrays,p):
 q=obj[0];items=[];errors=[]
 for a in arrays:
  assert not a['ownership'].any() and a['stochastic'].all() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
  assert np.array_equal(a['states'][1:],a['next_states'][:-1])
  x=torch.tensor(a['states']);raw=torch.tensor(a['raw_actions']);atom=torch.tensor(a['atom']);old=torch.tensor(a['logp'])
  with torch.no_grad():
   loc=q.sequence(x);me=float((loc-torch.tensor(a['locations'])).abs().max());le=float((P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max());assert me<1e-10 and le<1e-9,(me,le)
  items.append(dict(x=x,raw=raw,atom=atom,old=old));errors.append([me,le])
 if p['arm']=='terminal':
  z=np.asarray([float(a['end_flags'][-1,1] and not a['end_flags'][:,0].any()) for a in arrays]);ad=z-(z.sum()-z)/(len(z)-1)
  scale=float(np.sqrt(np.mean(ad**2)))
  for b,v in zip(items,ad):b['adv']=b['old'].new_full(b['old'].shape,float(v)/(scale+1e-8))
  diag=dict(outcomes=z.tolist(),LOO=ad.tolist(),coefficient_RMS=scale,homogeneous_outcomes_zero_RL_signal=bool(scale==0),density_errors=errors)
 else:
  credit,trace=Credit.peer_batch(obj,arrays,p)
  for b,c in zip(items,credit):b['adv']=c['adv']
  scale=max(float(b['adv'].abs().max()) for b in items);diag=dict(native_time_RTG=trace,homogeneous_outcomes_zero_RL_signal=bool(scale==0),density_errors=errors)
 return items,diag

def fit(obj,arrays,p,guard,event):
 q,opt=obj;items,diagnosis=batch(obj,arrays,p);refs=references(p);trace=[]
 if diagnosis['homogeneous_outcomes_zero_RL_signal']:return dict(steps=0,reason='ZERO_REGISTERED_RL_SIGNAL_NO_ARTIFICIAL_UPDATE',diagnosis=diagnosis,trace=[],trainable_parameters=sum(v.numel() for v in q.parameters() if v.requires_grad))
 for k in range(p['inner_steps']):
  guard();opt.zero_grad(set_to_none=True);rl=keep=0.
  for b in items:
   loc=q.sequence(b['x']);ratio=(P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();assert torch.isfinite(ratio).all()
   loss=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).sum()/len(items);loss.backward();rl+=float(loss.detach())
  # This term has zero gradient at an identical reference policy. Four actual
  # inner steps allow it to act; it is not a first-step safety mechanism.
  for ident,x,target in refs:
   loc=q.sequence(x);kl=.5*((loc-target)/loc.new_tensor(p['raw_sigma'])).square().sum(-1).mean()/len(refs)
   (p['replay_weight']*kl).backward();keep+=float(kl.detach())
  params=[v for v in q.parameters() if v.requires_grad];norm=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
  event('BEFORE',k);opt.step();event('AFTER',k);assert not opt.state
  with torch.no_grad():
   scores=[P.L.density(q.sequence(b['x']),b['raw'],b['atom'],p['raw_sigma'])-b['old'] for b in items]
   kl=float(sum((s.exp()-1-s).mean() for s in scores)/len(scores))
  trace.append(dict(step=k,RL_loss=rl,successful_history_KL=keep,gradient_norm=norm,sampled_KL=kl))
  if kl>p['target_kl']:break
 return dict(steps=len(trace),diagnosis=diagnosis,trace=trace,trainable_parameters=sum(v.numel() for v in q.parameters() if v.requires_grad),new_fresh_policy_batch=True,Value_updates=0,not_claimed_unbiased_or_original=True)

def export(obj,p,path):
 s=state(obj);buf=io.BytesIO();torch.save(dict(joint_state=s,policy_hash=obj[0].phash(),joint_hash=digest(s),architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,source_immutable_Value_and_Adam=p['source_initial'],optimizer='FRESH_NO_MOMENTUM_SGD'),buf)
 b=gzip.compress(buf.getvalue(),mtime=0);assert len(b)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(b);f.flush();__import__('os').fsync(f.fileno())
 assert A.frozen(p,path).phash()==obj[0].phash()
 return dict(path=str(path),sha256=sha(path),bytes=len(b),policy_hash=obj[0].phash(),joint_hash=digest(s))
