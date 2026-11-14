"""Verified real feedback labels + maximally interfered recurrent rehearsal.

One-factor prefix-preservation ablation; no new policy gradients or novelty
claim. Executed deterministic teacher labels are never used as RL samples.
"""
from pathlib import Path
import copy,gzip,io,math,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t542_offline_field_relay'));import t542_learning as Old
A,P,G,torch,np=Old.A,Old.P,Old.G,Old.torch,Old.np
read,save,sha,digest=Old.read,Old.save,Old.sha,Old.digest

def make(p):
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 q=A.frozen(p,p['initial']['path']).train().requires_grad_(True);opt=torch.optim.Adam(q.parameters(),lr=p['repair_lr'],foreach=False,fused=False);assert not opt.state
 return q,opt

def state(obj):return dict(actor=copy.deepcopy(obj[0].state_dict()),actor_opt=copy.deepcopy(obj[1].state_dict()))
def load(obj,s):obj[0].load_state_dict(s['actor']);obj[1].load_state_dict(s['actor_opt'])

def references(rows):
 result=[]
 for row in rows:
  assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
  result.append(dict(id=row['job_id'],family=int(row['job_id'].startswith('t502_')),x=torch.tensor(a['states']),target=torch.tensor(a['locations'])))
 assert [sum(r['family']==i for r in result) for i in [0,1]]==[7,12]
 return result

def loss_parts(q,a,p):
 x=torch.tensor(a['states']);mask=torch.tensor(a['ownership'].astype(bool));y=torch.tensor(a['teacher_actions'],dtype=torch.float64);assert mask.any() and not a['stochastic'].any()
 loc=q.sequence(x);err=P.action_loss(loc[mask],y[mask]).mean(-1)
 tail=.5*err.mean()+.5*err.topk(max(1,math.ceil(.2*len(err)))).values.mean()
 prefix=~mask;assert prefix.any() and not mask[:int(prefix.sum())].any() and mask[int(prefix.sum()):].all()
 target=torch.tensor(a['locations']);preserve=.5*((loc[prefix]-target[prefix])/loc.new_tensor(p['raw_sigma'])).square().sum(-1).mean()
 return tail,preserve

def repair(q,a,p):
 tail,prefix=loss_parts(q,a,p);return tail+p['prefix_keep_weight']*prefix

def memory(q,g,p):
 loc=q.sequence(g['x']);return .5*((loc-g['target'])/loc.new_tensor(p['raw_sigma'])).square().sum(-1).mean()

def ranking(obj,a,refs,p):
 q,opt=obj;before=state(obj)
 with torch.no_grad():pre=[float(memory(q,g,p)) for g in refs]
 opt.zero_grad(set_to_none=True);r=repair(q,a,p);r.backward();norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));opt.step()
 with torch.no_grad():post=[float(memory(q,g,p)) for g in refs]
 load(obj,before);assert digest(state(obj))==digest(before)
 measures=[dict(id=g['id'],family=g['family'],loss_before=b,loss_after=v,interference_increase=v-b) for g,b,v in zip(refs,pre,post)]
 selected=[]
 for family,number in [(0,2),(1,3)]:selected.extend(sorted([m for m in measures if m['family']==family],key=lambda m:(-m['interference_increase'],m['id']))[:number])
 assert len(selected)==5
 return [r['id'] for r in selected],dict(measures=measures,selected=[r['id'] for r in selected],virtual_discarded_Adam=1,virtual_repair_loss=float(r.detach()),virtual_gradient_norm=norm)

def fit(obj,a,p,refs,guard,event):
 q,opt=obj;trace=[];lookup={g['id']:g for g in refs}
 for k in range(p['fit_steps']):
  guard();event('VIRTUAL_BEFORE',k);chosen,rank=ranking(obj,a,refs,p);event('VIRTUAL_ROLLED_BACK',k)
  opt.zero_grad(set_to_none=True)
  with torch.no_grad():tail,prefix=loss_parts(q,a,p);tail_value=float(tail);prefix_value=float(prefix)
  rl=repair(q,a,p);(.5*rl).backward();rehearsal=0.
  for family,weight in [(0,.25),(1,.25)]:
   pool=[lookup[i] for i in chosen if lookup[i]['family']==family]
   for g in pool:
    z=memory(q,g,p);(weight*z/len(pool)).backward();rehearsal+=weight*float(z.detach())
  norm=float(torch.nn.utils.clip_grad_norm_(q.parameters(),1.,error_if_nonfinite=True));event('BEFORE',k);opt.step();event('AFTER',k)
  trace.append(dict(step=k,recovery_loss=float(rl.detach()),tail_executed_action_loss=tail_value,protected_prefix_KL=prefix_value,selected_rehearsal_loss=rehearsal,gradient_norm=norm,ranking=rank))
 assert all(int(opt.state[z]['step'])==p['fit_steps'] for z in q.parameters())
 return dict(Actor_Adam=p['fit_steps'],virtual_discarded_Adam=p['fit_steps'],Value_updates=0,new_policy_gradient_updates=0,trace=trace,current_student_feedback_states=True,teacher_executed_action_feasible_set=True,explicit_student_prefix_location_preservation=True,prefix_keep_weight=p['prefix_keep_weight'],full_history_recomputed=True,standard_DAgger_MIR_inspired_NOT_original=True)

def export(obj,p,path):
 s=state(obj);b=io.BytesIO();torch.save(dict(joint_state=s,policy_hash=obj[0].phash(),joint_hash=digest(s),architecture='T501_single120_64carry_two_neural_command_residuals',single_actor=True,source_immutable_Value_and_Adam=p['initial'],optimizer='FRESH_FULL_ACTOR_ADAM'),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as h:h.write(blob);h.flush();__import__('os').fsync(h.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=obj[0].phash(),joint_hash=digest(s))
