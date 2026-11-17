"""Stratified fresh PG and successful-recovery supervision, no runtime teacher."""
from pathlib import Path
import sys,os,copy,io,gzip
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;DOC=ROOT/'stage/docs/overnight_20261117';OUT=ROOT/'stage/runs/t629_offline_night_relay'
sys.path.insert(0,str(ROOT/'stage/experiments/t628s_actual_memory_recovery'))
import t628s_learning as Prior
Prior.ARCH='T629_T608_branch_sigma_explicit'
L,torch,np=Prior.L,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
state,load,frozen,_prior_export,make,set_sigma,configure_gpu=Prior.state,Prior.load,Prior.frozen,Prior.export,Prior.make,Prior.set_sigma,Prior.configure_gpu

def export(obj,p,path):return _prior_export(obj,dict(p,student_initial=p['reference_model']),path)

def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);return c

def references(q,p):
 pc=dict(p,student_initial=p['reference_model']);return Prior.references(q,pc)

def batch(q,groups,p):
 items=[];diag=[]
 for group in groups:
  assert len(group)==2;parts=[]
  for a in group:
   assert a['stochastic'].all() and not a['ownership'].any() and not a['end_flags'][:-1].any() and a['end_flags'][-1].sum()==1
   x=torch.tensor(a['states'],device='cuda');raw=torch.tensor(a['raw_actions'],device='cuda');atom=torch.tensor(a['atom'],device='cuda');old=torch.tensor(a['logp'],device='cuda')
   with torch.no_grad():
    loc=q.sequence(x);assert float((loc-torch.tensor(a['locations'],device='cuda')).abs().max())<1e-10;assert float((L.P.L.density(loc,raw,atom,p['raw_sigma'])-old).abs().max())<1e-9
   dt=a['native_counts'].astype(np.int64);t=np.r_[0,np.cumsum(dt)[:-1]];ret=np.empty(len(dt));v=0.
   for k in range(len(dt)-1,-1,-1):v=float(a['rewards'][k])+p['gamma_native']**int(dt[k])*v;ret[k]=v
   parts.append(dict(x=x,raw=raw,atom=atom,old=old,initial=loc.detach(),time=t,end=int(dt.sum()),rtg=ret))
  for i,b in enumerate(parts):
   peer=parts[1-i];idx=np.searchsorted(peer['time'],b['time'],side='right')-1;valid=b['time']<peer['end'];v=np.zeros(len(b['time']));v[valid]=peer['rtg'][idx[valid]];b['adv0']=p['gamma_native']**b['time']*(b['rtg']-v)
  scale=np.sqrt(sum(float(np.mean(b['adv0']**2)) for b in parts)/2)+1e-8
  for b in parts:b['adv']=b['old'].new_tensor(b['adv0']/scale)
  items.append(parts);diag.append(dict(returns=[float(b['rtg'][0]) for b in parts],peer_RMS=float(scale)))
 return items,diag

def pg_objective(q,items,refs,p,weights):
 loss=next(q.branch.parameters()).new_zeros(());pg=0.;keep=0.
 assert len(weights)==len(items) and abs(sum(weights)-1)<1e-12
 for weight,group in zip(weights,items):
  for b in group:
   loc=q.sequence(b['x']);ratio=(L.P.L.density(loc,b['raw'],b['atom'],p['raw_sigma'])-b['old']).exp();value=-torch.minimum(ratio*b['adv'],ratio.clamp(.8,1.2)*b['adv']).mean()*weight/len(group);loss=loss+value;pg+=float(value.detach())
 for r in refs:
  loc=q.sequence(r['x'])[r['mask']];target=r['target'][r['mask']];value=.5*((loc-target)/loc.new_tensor(p['retention_sigma'])).square().mean()/len(refs);loss=loss+value*p['keep_weight'];keep+=float(value.detach())
 return loss,dict(policy_loss=pg,keep=keep)

def fit_pg(obj,groups,p,weights,guard,event):
 q,opt=obj;items,diag=batch(q,groups,p);refs=references(q,p);body=L.fixed(q);trace=[]
 for k in range(p['actor_steps']):
  guard();before=state(obj);opt.zero_grad(set_to_none=True);loss,detail=pg_objective(q,items,refs,p,weights);loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);kl=Prior.P.trust(q,items,p);okay=np.isfinite(kl).all() and max(kl)<=p['max_task_latent_KL']
  if not okay:load(obj,before)
  trace.append(dict(step=k,loss=float(loss.detach()),gradient_norm=norm,task_latent_KL=kl,accepted=bool(okay),**detail))
  if not okay:break
 assert L.fixed(q)==body
 return dict(attempted_optimizer=len(trace),applied=sum(t['accepted'] for t in trace),trace=trace,peer_batch=diag,weights=weights,full_current_history_BPTT=True,no_Critic=True)

def recovery_data(obj,p,corrections,anchors):
 q=obj[0];data=[]
 for a,start,model in corrections:
  teacher=frozen(p,model['path']).cuda();x=torch.tensor(a['states'],device='cuda');mask=torch.arange(len(x),device='cuda')>=start
  with torch.no_grad():target=teacher.sequence(x).detach()
  assert np.array_equal(target[start:].clamp(-1,1).cpu().float().numpy(),a['actions'][start:]);data.append(dict(x=x,mask=mask,target=target,kind='qualified_recovery'))
 for a,model in anchors:
  teacher=frozen(p,model['path']).cuda();x=torch.tensor(a['states'],device='cuda')
  with torch.no_grad():target=teacher.sequence(x).detach()
  assert np.array_equal(target.clamp(-1,1).cpu().float().numpy(),a['actions']);data.append(dict(x=x,mask=torch.ones(len(x),dtype=torch.bool,device='cuda'),target=target,kind='new_success_anchor'))
 assert any(r['kind']=='qualified_recovery' for r in data) and len(anchors)==2
 return data

def fit_recovery(obj,corrections,anchors,p,guard,event):
 q,opt=obj;data=recovery_data(obj,p,corrections,anchors);refs=references(q,p);body=L.fixed(q);trace=[]
 for k in range(p['recovery_steps']):
  guard();before=state(obj);opt.zero_grad(set_to_none=True);loss=next(q.branch.parameters()).new_zeros(());terms=[]
  for r in data:
   loc=q.sequence(r['x'])[r['mask']];target=r['target'][r['mask']];v=((loc-target)/loc.new_tensor(p['retention_sigma'])).square().mean()/len(data);loss=loss+v;terms.append(float(v.detach()))
  for r in refs:
   loc=q.sequence(r['x'])[r['mask']];v=.5*((loc-r['target'][r['mask']])/loc.new_tensor(p['retention_sigma'])).square().mean()/len(refs);loss=loss+p['keep_weight']*v
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(L.variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
  with torch.no_grad():finite=all(torch.isfinite(v).all() for v in L.variables(q))
  if not finite:load(obj,before)
  trace.append(dict(step=k,loss=float(loss.detach()),terms=terms,gradient_norm=norm,accepted=bool(finite)))
  if not finite:break
 assert L.fixed(q)==body
 return dict(attempted_optimizer=len(trace),applied=sum(t['accepted'] for t in trace),trace=trace,teacher_success_verified_before_labels=True,only_teacher_suffix_supervised=True,no_teacher_hidden_transfer=True,no_RL_from_mixed_history=True)
