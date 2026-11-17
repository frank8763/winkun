"""Finite GEM-inspired actual-Adam displacement projection, not a safety proof."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t606r_offline_retention_recovery'))
import t606r_learning as Previous
Prior,Old,P,G,torch,np=Previous.Prior,Previous.Old,Previous.P,Previous.G,Previous.torch,Previous.np
read,save,sha,digest=Previous.read,Previous.save,Previous.sha,Previous.digest
make,dataset,frozen,variables,fixed,state,load,export=Previous.make,Previous.dataset,Previous.frozen,Previous.variables,Previous.fixed,Previous.state,Previous.load,Previous.export
Credit=Previous.Credit

def task_losses(q,data,packed):
 pred=q.branch(packed)[0];out=[]
 for i,g in enumerate(data):
  loc=(g['base']+pred[:len(g['n']),i]-g['reference'])[g['mask']]
  per=P.action_loss(loc,g['target'][g['mask']].clamp(-1,1)).mean(-1)
  out.append(.5*per.mean()+.5*per.topk(max(1,math.ceil(.1*len(per)))).values.mean())
 return torch.stack(out)

def project(delta,normal,max_passes=128,tolerance=1e-10):
 """Dykstra projection onto finitely many normalized closed half-spaces."""
 if not len(normal):return delta.clone(),dict(passes=0,max_violation=0.,zero_normals=True)
 norm=torch.linalg.vector_norm(normal,dim=1);normal=normal[norm>1e-14];norm=norm[norm>1e-14]
 if not len(normal):return delta.clone(),dict(passes=0,max_violation=0.,zero_normals=True)
 normal=normal/norm[:,None];x=delta.clone();correction=torch.zeros_like(normal)
 for it in range(max_passes):
  old=x.clone()
  for i,g in enumerate(normal):
   y=x+correction[i];z=y-torch.clamp(torch.dot(g,y),min=0)*g;correction[i]=y-z;x=z
  violation=float((normal@x).max())
  if violation<=tolerance and float((x-old).abs().max())<=tolerance:break
 return x,dict(passes=it+1,max_violation=max(0.,violation),zero_normals=False,converged=violation<=tolerance)

def flat(values):return torch.cat([v.reshape(-1) for v in values])

def fit(obj,data,p,guard,event,arrays=None,steps=None):
 assert arrays is None and p['arm']=='command';q,opt=obj;body=fixed(q);vs=variables(q)
 packed=torch.nn.utils.rnn.pad_sequence([g['n'] for g in data]);indices=[i for i,g in enumerate(data) if g['kind']=='old' and g['id'] in p['teacher_initial_success_ids']]
 assert len(indices)==31 and len(set(data[i]['id'] for i in indices))==31
 before=Old.measure(q,data);trace=[];applied=0
 for k in range(p['fit_steps'] if steps is None else steps):
  guard();prior=state(obj);opt.zero_grad(set_to_none=True);tasks=task_losses(q,data,packed);loss=.5*tasks.mean()+.5*tasks.topk(10).values.mean();value=float(loss.detach());protected=tasks[indices].detach().clone()
  normal=torch.stack([flat(torch.autograd.grad(tasks[i],vs,retain_graph=True)) for i in indices]).detach()
  loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(vs,1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k);proposal=state(obj)
  names=[name for name,v in q.named_parameters() if name.startswith('branch.')]
  delta=flat([proposal['actor'][name]-prior['actor'][name] for name in names]);projected,diagnostic=project(delta,normal)
  trials=[];accepted=False
  for factor in p['step_factors']:
   guard()
   with torch.no_grad():
    offset=0
    for name in names:
     v=dict(q.named_parameters())[name];n=v.numel();v.copy_(prior['actor'][name]+factor*projected[offset:offset+n].reshape_as(v));offset+=n
    nxt=task_losses(q,data,packed);nloss=.5*nxt.mean()+.5*nxt.topk(10).values.mean();raw,cmd=Previous.protection(q,data,packed)
   excess=float((nxt[indices]-protected).max());okay=bool(torch.isfinite(nloss)) and float(nloss)<=value+1e-12 and excess<=1e-10 and cmd<=p['protected_max_change'] and diagnostic.get('converged',True)
   trials.append(dict(factor=factor,loss=float(nloss),raw_max=raw,command_max=cmd,protected_task_max_increase=excess,accepted=okay))
   if okay:accepted=True;applied+=1;break
  if not accepted:load(obj,prior)
  trace.append(dict(step=k,before_loss=value,gradient_norm=norm,projection=diagnostic,unprojected_norm=float(torch.linalg.vector_norm(delta)),projected_norm=float(torch.linalg.vector_norm(projected)),trials=trials,applied=accepted))
  if not accepted:break
 assert fixed(q)==body
 return dict(attempted_Adam=len(trace),applied_parameter_steps=applied,trace=trace,before=before,after=Old.measure(q,data),fixed_sha=body,peer_batch=None,new_RL_attempted_Adam=0,new_Critic=0,full_history_BPTT=True,actual_Adam_displacement_projection=True,protected_task_ids=[data[i]['id'] for i in indices],protected_histories_are_T550_not_current_student_visits=True,scaled_Adam_moments_from_full_proposal=True,rejected_entire_step_restored=True,closed_loop_guarantee=False)
