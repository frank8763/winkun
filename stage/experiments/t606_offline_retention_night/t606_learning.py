"""Fixed-horizon retention pilots. Supervised arms and separately labelled fresh PG.

The line search is a finite empirical function constraint, not a safety proof.
Adam moments come from the full proposal when a scaled parameter step is kept;
rejection restores both Actor and Adam. All such attempts are counted.
"""
from pathlib import Path
import sys,copy,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t604_reverse_initialization'))
import t604_learning as Prior
Old=Prior.Old
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
state,load,fixed,variables,frozen,export=Prior.state,Prior.load,Prior.fixed,Prior.variables,Prior.frozen,Prior.export
dataset=Prior.dataset
Credit=Old.Previous.Credit

def make(p):
 m=p['student_initial'];assert sha(m['path'])==m['sha256'];q=frozen(p,m['path']);assert q.phash()==m['policy_hash'];q.requires_grad_(False);q.branch.requires_grad_(True)
 opt=torch.optim.Adam(variables(q),lr=p['fit_lr'],foreach=False,fused=False)
 if p.get('resume_optimizer',False):opt.load_state_dict(P.checkpoint(m['path'])['actor_opt'])
 for g in opt.param_groups:g['lr']=p['rl_lr'] if p['arm']=='rl' else p['fit_lr']
 return q,opt

def risks(q,data,packed,domain):
 pred=q.branch(packed)[0];tasks=[]
 for i,g in enumerate(data):
  loc=(g['base']+pred[:len(g['n']),i]-g['reference'])[g['mask']];t=g['target'][g['mask']]
  per=(loc-t).square().mean(-1) if domain=='raw' else P.action_loss(loc,t.clamp(-1,1)).mean(-1)
  tasks.append(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())
 tasks=torch.stack(tasks);return .5*tasks.mean()+.5*tasks.topk(10).values.mean()

def protection(q,data,packed):
 pred=q.branch(packed)[0];raw=pred.new_zeros(());command=pred.new_zeros(())
 for i,g in enumerate(data):
  if g['kind']!='recovery':continue
  loc=(g['base']+pred[:len(g['n']),i]-g['reference'])[g['mask']];target=g['target'][g['mask']]
  raw=torch.maximum(raw,(loc-target).abs().max());command=torch.maximum(command,(loc.clamp(-1,1)-target.clamp(-1,1)).abs().max())
 return float(raw),float(command)

def objective(q,data,packed,p,items):
 aux=risks(q,data,packed,'raw' if p['arm']=='raw' else 'command');loss=aux;kl=0.
 if items is not None:
  pg=aux.new_zeros(());kls=[]
  for b in items:
   loc=q.sequence(b['x']);z=P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old'];ratio=z.exp()
   pg-=torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()/len(items)
   kls.append((ratio-1-z).mean())
  loss=pg+aux;kl=float(torch.stack(kls).mean().detach())
 return loss,dict(aux=float(aux.detach()),KL=kl)

def fit(obj,data,p,guard,event,arrays=None,steps=None):
 q,opt=obj;body=fixed(q);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);items=diag=None
 if p['arm']=='rl':
  assert arrays is not None and len(arrays)==4
  items,diag=Credit.peer_batch(obj,arrays,p)
 before=Old.measure(q,data);trace=[];applied=0
 for k in range((p['rl_steps'] if p['arm']=='rl' else p['fit_steps']) if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed,p,items);value=float(loss.detach());loss.backward()
  norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
  proposal=state(obj);trials=[];accepted=False
  for factor in p['step_factors']:
   guard()
   with torch.no_grad():
    for name,v in q.named_parameters():
     if name.startswith('branch.'):v.copy_(prior['actor'][name]+factor*(proposal['actor'][name]-prior['actor'][name]))
    nxt,detail=objective(q,data,packed,p,items);raw,cmd=protection(q,data,packed)
   drift=raw if p['arm']=='raw' else cmd
   okay=math.isfinite(float(nxt)) and float(nxt)<=value+1e-12 and drift<=p['protected_max_change'] and detail['KL']<=p['target_kl']
   trials.append(dict(factor=factor,loss=float(nxt),raw_max=raw,command_max=cmd,KL=detail['KL'],accepted=okay))
   if okay:accepted=True;applied+=1;break
  if not accepted:load(obj,prior)
  trace.append(dict(step=k,before_loss=value,before_parts=parts,gradient_norm=norm,trials=trials,applied=accepted))
  if not accepted:break
 assert fixed(q)==body
 return dict(attempted_Adam=len(trace),applied_parameter_steps=applied,trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,peer_batch=diag,
  new_RL_attempted_Adam=len(trace) if p['arm']=='rl' else 0,new_Critic=0,full_history_BPTT=True,not_standard_value_GAE_PPO=True,
  scaled_Adam_moments_from_full_proposal=True,rejected_entire_step_restored=True)
