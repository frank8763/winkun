"""One finite supervised recovery proposal using a genuinely qualified teacher suffix."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'))
import t608_learning as Previous
Prior,Old,P,G,torch,np=Previous.Prior,Previous.Old,Previous.P,Previous.G,Previous.torch,Previous.np
read,save,sha,digest=Previous.read,Previous.save,Previous.sha,Previous.digest
state,load,fixed,variables,frozen,export,make=Previous.state,Previous.load,Previous.fixed,Previous.variables,Previous.frozen,Previous.export,Previous.make
Credit=Previous.Credit

def dataset(p):
 data=Previous.dataset(p);r=read(p['qualified_recovery_manifest']);assert r['state']=='QUALIFIED_SUCCESSFUL_TRAINING_ONLY_RECOVERY_DATA' and r['success'] and not r['collision'] and sha(r['path'])==r['sha256'];a=G.E.N.S.arrays(r['path']);assert all(digest(v)==r['native_array_hashes'][k] for k,v in a.items());k=r['teacher_start_decision'];mask=torch.tensor(a['ownership'].astype(bool));assert not mask[:k].any() and mask[k:].all() and len(mask)>k
 q=Old.initial(p);teacher=frozen(p,r['teacher']['path']);x=torch.tensor(a['states']);target=torch.tensor(a['teacher_locations'])
 with torch.no_grad():
  assert float((teacher.sequence(x)-target).abs().max())<1e-12 and np.array_equal(target[k:].clamp(-1,1).float().numpy(),a['actions'][k:]);n=q.normalized(x);base=Old.Parent.Actor.sequence(q,x).detach();reference=q.anchor(n[:,None])[0][:,0].detach()
 data.append(dict(id='qualified_actual_prefix_'+r['task'],kind='correction',x=x,n=n,base=base,reference=reference,target=target,mask=mask,source='T612_SUCCESSFUL_T550_SUFFIX_ONLY'))
 assert len(data)==43 and len(data[-1]['x'])==len(mask);return data

def objective(q,data,packed,p):
 pred=q.branch(packed)[0];parts=[]
 for i,g in enumerate(data):
  loc=(g['base']+pred[:len(g['n']),i]-g['reference'])[g['mask']];target=g['target'][g['mask']].clamp(-1,1);per=P.action_loss(loc,target).mean(-1);parts.append(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())
 old=torch.stack(parts[:42]);replay=.5*old.mean()+.5*old.topk(10).values.mean();correction=parts[42];loss=replay+correction
 return loss,dict(replay=float(replay.detach()),correction=float(correction.detach()),KL=0.)

def protection(q,data,packed):return Previous.protection(q,data,packed)

def fit(obj,data,p,guard,event,arrays=None,steps=None):
 assert arrays is None and p['arm']=='command';q,opt=obj;body=fixed(q);packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);before=Old.measure(q,data);trace=[];applied=0
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);loss,parts=objective(q,data,packed,p);value=float(loss.detach());loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(variables(q),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj);trials=[];accepted=False
  for factor in p['step_factors']:
   guard()
   with torch.no_grad():
    for name,v in q.named_parameters():
     if name.startswith('branch.'):v.copy_(prior['actor'][name]+factor*(proposal['actor'][name]-prior['actor'][name]))
    nxt,detail=objective(q,data,packed,p);raw,cmd=protection(q,data,packed)
   okay=math.isfinite(float(nxt)) and float(nxt)<=value+1e-12 and cmd<=p['protected_max_change']
   trials.append(dict(factor=factor,loss=float(nxt),raw_max=raw,command_max=cmd,parts=detail,accepted=okay))
   if okay:accepted=True;applied+=1;break
  if not accepted:load(obj,prior)
  trace.append(dict(step=k,before_loss=value,before_parts=parts,gradient_norm=norm,trials=trials,applied=accepted))
  if not accepted:break
 assert fixed(q)==body
 return dict(attempted_Adam=len(trace),applied_parameter_steps=applied,trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,new_RL_attempted_Adam=0,new_BC_attempted_Adam=len(trace),new_Critic=0,full_history_BPTT=True,correction_mask_is_teacher_suffix_ownership_true=True,teacher_hidden_never_used_as_student_hidden=True,scaled_Adam_moments_from_full_proposal=True,rejected_entire_step_restored=True)
