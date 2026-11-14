"""Preserve the T546 forward arithmetic, add one separate neural residual."""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t546_functional_residual'));import t546_actor as Old
P,G,torch,np=Old.P,Old.G,Old.torch,Old.np
read,save,sha,digest=Old.read,Old.save,Old.sha,Old.digest
ARCH='T548_single120_64carry_immutableT546_plus_protected_compact_residual'

class Actor(Old.Actor):
 def __init__(self,p,old_n,new_n):
  super().__init__(p,old_n);self.register_buffer('added_centers',torch.zeros(new_n,184,dtype=torch.float64));self.register_buffer('added_radii',torch.ones(new_n,dtype=torch.float64));self.added_weights=torch.nn.Parameter(torch.zeros(new_n,2,dtype=torch.float64));self.requires_grad_(False)
 def location(self,h,n):
  original=super().location(h,n);phi=Old.basis(torch.cat([h,n],-1),self.added_centers,self.added_radii);residual=phi@self.added_weights
  return torch.where((phi!=0).any(-1,keepdim=True),original+residual,original)
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Old.frozen(p,path)
 state=s['joint_state']['actor'];q=Actor(p,len(state['repair_centers']),len(state['added_centers']));q.load_state_dict(state);assert q.phash()==s['policy_hash'];return q.eval()

def fit(p,a,event=lambda *_:None):
 q0=Old.frozen(p,p['initial']['path']);owner=a['ownership'].astype(bool);assert owner.any() and np.flatnonzero(owner)[0]==p['teacher_first'];assert int(owner.sum())<=p['maximum_new_centers']
 protected=[]
 with torch.no_grad():
  for row in p['protected20_rows']:
   assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];data=G.E.N.S.arrays(row['path']);x=torch.tensor(data['states']);protected.append(Old.features(q0,x))
  x=torch.tensor(a['states']);features=Old.features(q0,x);protected.append(features[~owner]);points=torch.cat(protected);centers=features[owner].clone();nearest=centers.new_full((len(centers),),float('inf'))
  for offset in range(0,len(points),256):nearest=torch.minimum(nearest,((centers[:,None]-points[offset:offset+256])**2).sum(-1).min(-1).values)
  distances=nearest.sqrt();assert torch.isfinite(distances).all() and float(distances.min())>p['minimum_feature_separation'];radii=p['support_fraction']*distances
  assert all(float(Old.basis(points[k:k+256],centers,radii).max())==0. for k in range(0,len(points),256));K=Old.basis(centers,centers,radii);condition=float(np.linalg.cond(K.numpy()));assert condition<=p['maximum_condition'];event('BEFORE_CLOSED_FORM',dict(centers=len(centers),condition=condition,minimum_separation=float(distances.min())))
  original=q0.sequence(x);target=torch.tensor(a['teacher_actions'],dtype=torch.float64)[owner];weights=torch.tensor(np.linalg.solve(K.numpy(),(target-original[owner]).numpy()));event('AFTER_CLOSED_FORM',dict(closed_form_solves=1))
  q=Actor(p,len(q0.repair_centers),len(centers));missing,extra=q.load_state_dict(q0.state_dict(),strict=False);assert set(missing)=={'added_centers','added_radii','added_weights'} and not extra
  q.added_centers.copy_(centers);q.added_radii.copy_(radii);q.added_weights.copy_(weights);pred=q.sequence(x);assert np.array_equal(pred.clamp(-1,1).float().numpy(),a['actions']);assert torch.equal(pred[~owner],original[~owner])
  for row in p['protected20_rows']:
   data=G.E.N.S.arrays(row['path']);z=torch.tensor(data['states']);assert torch.equal(q.sequence(z),q0.sequence(z))
  for key,value in q0.state_dict().items():assert torch.equal(q.state_dict()[key],value)
  q.sample=False;q.reset_noise();hidden=None;previous=None
  for i,state in enumerate(x):
   command,loc,hidden,_=q.step(state[None],hidden);assert np.array_equal(command[0].numpy(),a['actions'][i]);assert np.array_equal(np.zeros(64) if i==0 else previous,a['hidden_before'][i]);previous=hidden[0,0].numpy().copy()
  q.reset_noise()
 return q,dict(closed_form_solves=1,new_Adam=0,new_RL=0,new_centers=len(centers),condition=condition,minimum_protected_feature_distance=float(distances.min()),all20_saved_function_outputs_bitwise_equal=True,originalT546_all_parameters_and_buffers_exact=True,teacher_actions_entire_history_float32_step_exact=True,source_grammar='Two fixed neural basis readouts; no runtime teacher/map/route/taskID gate',finite_support_only_NOT_robustness_or_global_safety=True)

def export(q,p,path):
 state=dict(actor=copy.deepcopy(q.state_dict()));buffer=io.BytesIO();torch.save(dict(joint_state=state,architecture=ARCH,policy_hash=q.phash(),joint_hash=digest(state),source_immutable_model=p['initial'],fit='ONE_FIXED_ADDITIONAL_NEURAL_READOUT_LINEAR_SOLVE',optimizer=None),buffer);blob=gzip.compress(buffer.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(state))
