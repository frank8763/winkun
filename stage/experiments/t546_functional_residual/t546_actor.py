"""One causal neural Actor: immutable learned GRU plus compact RBF readout.

Centers are frozen learned features, never world waypoints or a runtime
controller. Every basis is evaluated by the same neural forward graph.
Finite recorded-input preservation is checked, not a global safety claim.
"""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t501_joint_command_repair'));import t501_learning as Base
P,G,torch,np=Base.P,Base.G,Base.torch,Base.np
read,save,sha,digest=G.read,G.save,G.sha,G.digest
LEGACY_FROZEN=Base.frozen
ARCH='T546_single120_frozen64carry_compact_RBF_neural_residual'

def basis(features,centers,radii):
 d=((features[...,None,:]-centers)**2).sum(-1);return torch.relu(1-d/radii.square()).square()

class Actor(Base.Actor):
 def __init__(self,p,n):
  super().__init__(p);self.register_buffer('repair_centers',torch.zeros(n,184,dtype=torch.float64));self.register_buffer('repair_radii',torch.ones(n,dtype=torch.float64));self.repair_weights=torch.nn.Parameter(torch.zeros(n,2,dtype=torch.float64));self.requires_grad_(False)
 def location(self,h,n):
  original=super().location(h,n);f=torch.cat([h,n],-1);phi=basis(f,self.repair_centers,self.repair_radii);correction=phi@self.repair_weights
  # Exact old arithmetic on protected rows; active rows use a learned residual.
  return torch.where((phi!=0).any(-1,keepdim=True),original+correction,original)
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def frozen(p,path):
 saved=P.checkpoint(path)
 if saved.get('architecture')!=ARCH:return LEGACY_FROZEN(p,path)
 state=saved['joint_state']['actor'];q=Actor(p,len(state['repair_centers']));q.load_state_dict(state);assert q.phash()==saved['policy_hash'];return q.eval()

def features(q,x):
 n=q.normalized(x);h,_=q.core(q.encoder(n)[:,None]);return torch.cat([h[:,0],n],-1)

def fit(p,event=lambda *_:None):
 q0=LEGACY_FROZEN(p,p['initial']['path']);teacher=p['recovery_row'];a=G.E.N.S.arrays(teacher['path']);assert sha(teacher['path'])==teacher['sha256'];own=a['ownership'].astype(bool);assert np.flatnonzero(own)[0]==170
 protected=[]
 with torch.no_grad():
  for row in p['protected_rows']:
   assert row['success'] and not row['collision'] and sha(row['path'])==row['sha256'];data=G.E.N.S.arrays(row['path']);x=torch.tensor(data['states']);assert float((q0.sequence(x)-torch.tensor(data['locations'])).abs().max())<1e-10;protected.append(features(q0,x))
  x=torch.tensor(a['states']);f=features(q0,x);protected.append(f[~own]);old=torch.cat(protected);centers=f[own].clone();assert 0<len(centers)<=64
  # Fixed linear-feature branch: exact equality constraints on all stored rows.
  H=torch.cat([old,old.new_ones(len(old),1)],-1);_,s,vh=np.linalg.svd(H.numpy(),full_matrices=False);rank=int((s>s[0]*p['rank_rtol']).sum());nullity=H.shape[1]-rank
  if nullity:
   N=vh[rank:].T;repairH=torch.cat([centers,centers.new_ones(len(centers),1)],-1).numpy();linear_available=float(np.linalg.norm(repairH@N))
  else:linear_available=0.
  event('LINEAR_CONSTRAINT_RANK',dict(rank=rank,nullity=nullity,protected_rows=len(old),available_repair_norm=linear_available))
  nearest=centers.new_full((len(centers),),float('inf'))
  for off in range(0,len(old),256):nearest=torch.minimum(nearest,((centers[:,None]-old[off:off+256])**2).sum(-1).min(-1).values)
  distance=nearest.sqrt();assert torch.isfinite(distance).all() and float(distance.min())>p['minimum_feature_separation'];radii=p['support_fraction']*distance
  maximum=0.
  for off in range(0,len(old),256):maximum=max(maximum,float(basis(old[off:off+256],centers,radii).max()))
  assert maximum==0.;K=basis(centers,centers,radii);condition=float(np.linalg.cond(K.numpy()));assert condition<=p['maximum_condition']
  event('BEFORE_LINEAR_SOLVE',dict(closed_form_solves=0))
  original=q0.sequence(x);target=torch.tensor(a['teacher_actions'],dtype=torch.float64)[own];weights=torch.tensor(np.linalg.solve(K.numpy(),(target-original[own]).numpy()))
  event('COMPACT_MATRIX',dict(closed_form_solves=1,minimum_separation=float(distance.min()),condition=condition,centers=len(centers)))
  q=Actor(p,len(centers));missing,unexpected=q.load_state_dict(q0.state_dict(),strict=False);assert set(missing)=={'repair_centers','repair_radii','repair_weights'} and not unexpected
  q.repair_centers.copy_(centers);q.repair_radii.copy_(radii);q.repair_weights.copy_(weights)
  predicted=q.sequence(x);commands=predicted.clamp(-1,1).float().numpy();assert np.array_equal(commands,a['actions']),float(np.abs(commands-a['actions']).max())
  assert torch.equal(predicted[~own],original[~own])
  for row in p['protected_rows']:
   data=G.E.N.S.arrays(row['path']);z=torch.tensor(data['states']);assert torch.equal(q.sequence(z),q0.sequence(z))
  # Same step(batch1) inference used by workers, not just batched fitting.
  hidden=None;q.sample=False
  for i,state in enumerate(x):
   command,loc,hidden,_=q.step(state[None],hidden);assert np.array_equal(command[0].numpy(),a['actions'][i]);assert np.array_equal(np.zeros(64) if i==0 else previous_hidden,a['hidden_before'][i]);previous_hidden=hidden[0,0].numpy().copy()
  q.reset_noise()
 return q,dict(linear_branch=dict(protected_rows=len(old),dimensions=H.shape[1],rank=rank,nullity=nullity,available_repair_norm=linear_available,not_global_infeasibility=True),compact_branch=dict(centers=len(centers),features=184,support_fraction=p['support_fraction'],minimum_protected_feature_distance=float(distance.min()),minimum_radius=float(radii.min()),maximum_radius=float(radii.max()),protected_max_basis=maximum,condition=condition,teacher_entire_executed_float32_exact=True,all19_success_location_outputs_bitwise_equal=True,all170_target_prefix_outputs_bitwise_equal=True,step_and_sequence_contract=True),new_optimizer_updates=0,closed_form_linear_solves=1,new_RL=0,Value=0,finite_saved_inputs_only_NOT_generalization_or_safety=True)

def export(q,p,path):
 state=dict(actor=copy.deepcopy(q.state_dict()));buf=io.BytesIO();torch.save(dict(joint_state=state,policy_hash=q.phash(),architecture=ARCH,optimizer=None,fit='ONE_CLOSED_FORM_COMPACT_RBF_READOUT',source_immutable_actor_and_Adam=p['initial'],single_actor=True),buf);blob=gzip.compress(buf.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(state))
