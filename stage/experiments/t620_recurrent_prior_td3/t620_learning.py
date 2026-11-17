"""Recurrent TD3-style pilot with balanced prior/online episode replay.

The current Actor function is preserved at initialization. Both critics have
their own trainable recurrent encoders. This is not standard SAC or full RLPD.
"""
from pathlib import Path
import sys,copy,io,gzip,math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'))
import t608_learning as Old
torch,np,P,G=Old.torch,Old.np,Old.P,Old.G
read,save,sha,digest=Old.read,Old.save,Old.sha,Old.digest
frozen,variables,fixed,export=Old.frozen,Old.variables,Old.fixed,Old.export

class Critic(torch.nn.Module):
 def __init__(self,p):
  super().__init__();self.register_buffer('mean',torch.tensor(p['input_mean'],dtype=torch.float64));self.register_buffer('std',torch.tensor(p['input_scale'],dtype=torch.float64));assert bool((self.std>0).all())
  self.encoder=torch.nn.Sequential(torch.nn.Linear(120,16),torch.nn.LayerNorm(16),torch.nn.Tanh());self.core=torch.nn.GRU(16,16);self.head=torch.nn.Sequential(torch.nn.Linear(18,32),torch.nn.LayerNorm(32),torch.nn.Tanh(),torch.nn.Linear(32,1));self.double()
 def features(self,x):return self.core(self.encoder(((x-self.mean)/self.std).clamp(-10,10)))[0]
 def values(self,h,a):return self.head(torch.cat([h,a],-1)).squeeze(-1)

def make(p):
 q=frozen(p,p['student_initial']['path']);q.requires_grad_(False);q.branch.requires_grad_(True)
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(p['learner_seed']);critics=torch.nn.ModuleList([Critic(p),Critic(p)])
 return dict(actor=q,actor_opt=torch.optim.Adam(variables(q),lr=p['actor_lr'],foreach=False,fused=False),critics=critics,critic_opt=torch.optim.Adam(critics.parameters(),lr=p['critic_lr'],foreach=False,fused=False),target_actor=copy.deepcopy(q).requires_grad_(False),target_critics=copy.deepcopy(critics).requires_grad_(False))

def state(obj):
 return dict(actor_branch=copy.deepcopy(obj['actor'].branch.state_dict()),actor_opt=copy.deepcopy(obj['actor_opt'].state_dict()),critics=copy.deepcopy(obj['critics'].state_dict()),critic_opt=copy.deepcopy(obj['critic_opt'].state_dict()),target_actor_branch=copy.deepcopy(obj['target_actor'].branch.state_dict()),target_critics=copy.deepcopy(obj['target_critics'].state_dict()))
def load(obj,s):
 obj['actor'].branch.load_state_dict(s['actor_branch']);obj['actor_opt'].load_state_dict(s['actor_opt']);obj['critics'].load_state_dict(s['critics']);obj['critic_opt'].load_state_dict(s['critic_opt']);obj['target_actor'].branch.load_state_dict(s['target_actor_branch']);obj['target_critics'].load_state_dict(s['target_critics'])

def episode(row,p):
 assert sha(row['path'])==row['sha256'];a=G.E.N.S.arrays(row['path']);n=len(a['states']);assert np.array_equal(a['next_states'][:-1],a['states'][1:]);assert a['end_flags'][-1].any() and not a['end_flags'][:-1].any()
 x=torch.tensor(np.concatenate([a['states'],a['next_states'][-1:]],axis=0));u=torch.tensor(a['actions'],dtype=torch.float64);r=torch.tensor(a['rewards']);dt=torch.tensor(a['native_counts'].astype(np.int64));terminal=torch.tensor(a['end_flags'].any(-1))
 assert len(x)==n+1 and u.shape==(n,2) and bool((dt>0).all())
 return dict(x=x,u=u,r=r,discount=p['gamma_native']**dt,bootstrap=(~terminal).double(),n=n,id=row['job_id'],path=row['path'],sha256=row['sha256'])

def dataset(p,online_rows):
 manifest=read(p['prior_manifest']);prior=[episode(r,p) for r in manifest['full_prior_episodes'] if r['training_eligibility']=='PRIOR_SCHEMA_AND_EXISTING_AUDIT_VERIFIED'];online=[episode(r,p) for r in online_rows]
 assert len(prior)>=40 and len(online)==3
 return prior,online

def online_accounting(rows):
 totals=dict(new_collection_physics=0,guide_prefix_physics=0,autonomous_suffix_physics=0,exact_replay_physics=0,episodes=[])
 for row in rows:
  a=G.E.N.S.arrays(row['path']);counts=a['native_counts'].astype(np.int64);guide=a['ownership'].astype(bool);total=int(counts.sum());forced=int(counts[guide].sum())
  assert total==row['physics']
  totals['new_collection_physics']+=total;totals['guide_prefix_physics']+=forced;totals['autonomous_suffix_physics']+=total-forced;totals['exact_replay_physics']+=total
  totals['episodes'].append(dict(path=row['path'],physics=total,guide_prefix_physics=forced,autonomous_suffix_physics=total-forced))
 return totals

def batch(data,p,rng):
 prior,online=data;selected=[prior[int(rng.integers(len(prior)))] for _ in range(2)]+[online[int(rng.integers(len(online)))] for _ in range(2)]
 x=torch.nn.utils.rnn.pad_sequence([z['x'] for z in selected]);indices=[]
 for i,z in enumerate(selected):indices.extend([(int(k),i) for k in rng.integers(0,z['n'],size=p['points_per_episode'])])
 t=torch.tensor([k for k,i in indices]);b=torch.tensor([i for k,i in indices]);u=torch.stack([selected[i]['u'][k] for k,i in indices]);r=torch.stack([selected[i]['r'][k] for k,i in indices]);d=torch.stack([selected[i]['discount'][k] for k,i in indices]);mask=torch.stack([selected[i]['bootstrap'][k] for k,i in indices]);return selected,x,t,b,u,r,d,mask

def losses(obj,reference,pack,p,generator,actor=False):
 selected,x,t,b,u,reward,discount,mask=pack
 if not actor:
  with torch.no_grad():
   action=obj['target_actor'].sequence(x).clamp(-1,1)[t+1,b];noise=torch.randn(action.shape,dtype=action.dtype,generator=generator)*p['target_noise'];action=(action+noise.clamp(-p['target_noise_clip'],p['target_noise_clip'])).clamp(-1,1)
   qs=[q.values(q.features(x)[t+1,b],action) for q in obj['target_critics']];target=reward+discount*mask*torch.minimum(*qs)
  values=[q.values(q.features(x)[t,b],u) for q in obj['critics']];loss=sum(torch.nn.functional.mse_loss(v,target) for v in values)
  return loss,dict(critic_loss=float(loss.detach()),target_min=float(target.min()),target_max=float(target.max()),sampled_prior_episodes=2,sampled_online_episodes=2)
 # Critic parameters are held fixed while preserving its action derivative.
 q=obj['critics'][0]
 with torch.no_grad():h=q.features(x)[t,b];anchor=reference.sequence(x).clamp(-1,1)[t,b]
 actions=obj['actor'].sequence(x).clamp(-1,1)[t,b];gain=-q.values(h,actions).mean();protected=torch.tensor([selected[int(i)]['id'] in p['before_success_ids'] for i in b],dtype=torch.bool)
 retain=(actions[protected]-anchor[protected]).square().mean() if bool(protected.any()) else gain.new_zeros(())
 loss=gain+p['retention_weight']*retain
 return loss,dict(actor_loss=float(loss.detach()),negative_Q=float(gain.detach()),old_function_MSE=float(retain.detach()),protected_sampled_coordinates=int(protected.sum())*2)

def fit(obj,data,p,guard,event):
 rng=np.random.default_rng(p['learner_seed']);generator=torch.Generator().manual_seed(p['learner_seed']+1);reference=frozen(p,p['student_initial']['path']);body=fixed(obj['actor']);trace=[];actor_steps=0
 for k in range(p['critic_steps']):
  guard();pack=batch(data,p,rng);obj['critic_opt'].zero_grad(set_to_none=True);loss,detail=losses(obj,reference,pack,p,generator);loss.backward();cn=float(torch.nn.utils.clip_grad_norm_(obj['critics'].parameters(),1.,error_if_nonfinite=True));event('critic','BEFORE',k);obj['critic_opt'].step();event('critic','AFTER',k);actor_detail=None
  if (k+1)%p['policy_delay']==0:
   obj['actor_opt'].zero_grad(set_to_none=True)
   for q in obj['critics']:q.requires_grad_(False)
   try:al,actor_detail=losses(obj,reference,pack,p,generator,True);al.backward();an=float(torch.nn.utils.clip_grad_norm_(variables(obj['actor']),1.,error_if_nonfinite=True));event('actor','BEFORE',actor_steps);obj['actor_opt'].step();event('actor','AFTER',actor_steps)
   finally:
    for q in obj['critics']:q.requires_grad_(True)
   actor_detail['gradient_norm']=an;actor_steps+=1
   with torch.no_grad():
    for target,current in zip(obj['target_actor'].branch.parameters(),obj['actor'].branch.parameters()):target.lerp_(current,p['target_tau'])
    for target,current in zip(obj['target_critics'].parameters(),obj['critics'].parameters()):target.lerp_(current,p['target_tau'])
  trace.append(dict(step=k,critic=detail,critic_gradient_norm=cn,actor=actor_detail))
 assert fixed(obj['actor'])==body and actor_steps==p['critic_steps']//p['policy_delay']
 return dict(critic_steps=len(trace),actor_steps=actor_steps,trace=trace,full_actual_prefix_recomputed=True,critic_encoders_are_separate_trainable_GRUs=True,new_teacher_labels=0,new_BC_updates=0,not_standard_SAC_or_full_RLPD=True,initial_fixed_body_sha=body)

def save_joint(obj,p,path):
 buf=io.BytesIO();torch.save(dict(state=state(obj),initial=p['student_initial'],state_sha=digest(state(obj)),fixed_body_sha=fixed(obj['actor'])),buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['joint_bytes']
 with Path(path).open('xb') as f:f.write(blob)
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),state_sha=digest(state(obj)))
