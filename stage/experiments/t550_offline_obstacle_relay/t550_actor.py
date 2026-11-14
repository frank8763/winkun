"""Append-only neural basis layers; unchanged earlier forward arithmetic.

Each checkpoint stores only its own learned layer and a SHA-locked parent.
All parent bytes remain local, and inference is one causal Actor, never a
teacher/model/task router. Narrow trajectory memory is not robust RL.
"""
from pathlib import Path
import copy,gzip,io,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t549_curvature_gated_transfer'));import t549_actor as Prior
P,G,torch,np=Prior.P,Prior.G,Prior.torch,Prior.np
read,save,sha,digest=Prior.read,Prior.save,Prior.sha,Prior.digest
ARCH='T550_single120_64carry_append_only_neural_basis_layers'
class Layer(torch.nn.Module):
 def __init__(self,n):
  super().__init__();self.register_buffer('centers',torch.zeros(n,184,dtype=torch.float64));self.register_buffer('radii',torch.ones(n,dtype=torch.float64));self.weights=torch.nn.Parameter(torch.zeros(n,2,dtype=torch.float64))
class Actor(Prior.Actor):
 def __init__(self,p,old_n,added_n,sizes):
  super().__init__(p,old_n,added_n);self.layers=torch.nn.ModuleList([Layer(n) for n in sizes]);self.requires_grad_(False)
 def location(self,h,n):
  original=super().location(h,n);shape=original.shape;features=torch.cat([h,n],-1).reshape(-1,184)
  for layer in self.layers:
   out=[];active=[]
   for k in range(0,len(features),32):
    phi=Prior.Old.basis(features[k:k+32],layer.centers,layer.radii);out.append(phi@layer.weights);active.append((phi!=0).any(-1))
   residual=torch.cat(out).reshape(shape);mask=torch.cat(active).reshape(shape[:-1])[...,None];original=torch.where(mask,original+residual,original)
  return original
 def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))
def extend(q0,p,n):
 sizes=[len(z.centers) for z in q0.layers] if isinstance(q0,Actor) else []
 q=Actor(p,len(q0.repair_centers),len(q0.added_centers),sizes+[n]);missing,extra=q.load_state_dict(q0.state_dict(),strict=False)
 assert set(missing)=={'layers.'+str(len(sizes))+'.'+k for k in ['centers','radii','weights']} and not extra
 return q

def frozen(p,path):
 s=P.checkpoint(path)
 if s.get('architecture')!=ARCH:return Prior.frozen(p,path)
 parent=s['parent'];assert sha(parent['path'])==parent['sha256'];q0=frozen(p,parent['path']);assert q0.phash()==parent['policy_hash']
 q=extend(q0,p,len(s['layer']['centers']));q.layers[-1].load_state_dict(s['layer']);assert q.phash()==s['policy_hash'] and digest(dict(actor=q.state_dict()))==s['joint_hash'];return q.eval()

def fit(p,a,event=lambda *_:None):
 q0=frozen(p,p['initial']['path']);own=a['ownership'].astype(bool);assert own.all() and 1<len(a['states'])<=840
 protected=[]
 with torch.no_grad():
  for row in p['protected_rows']:
   assert sha(row['path'])==row['sha256'] and row['success'] and not row['collision'];x=torch.tensor(G.E.N.S.arrays(row['path'])['states']);protected.append(Prior.Old.features(q0,x))
  x=torch.tensor(a['states']);centers=Prior.Old.features(q0,x);points=torch.cat(protected);nearest=centers.new_full((len(centers),),float('inf'))
  for k in range(0,len(points),128):nearest=torch.minimum(nearest,((centers[:,None]-points[k:k+128])**2).sum(-1).min(-1).values)
  dist=nearest.sqrt();assert torch.isfinite(dist).all() and float(dist.min())>1e-10,'INCOMPATIBLE_PROTECTED_FEATURE'
  others=centers.new_full((len(centers),),float('inf'))
  for k in range(0,len(centers),16):
   pair=((centers[k:k+16,None]-centers)**2).sum(-1);pair[torch.arange(len(pair)),torch.arange(k,k+len(pair))]=float('inf');others[k:k+len(pair)]=pair.min(-1).values
  od=others.sqrt();assert torch.isfinite(od).all() and float(od.min())>1e-10,'DUPLICATE_TEACHER_FEATURE'
  radii=.49*torch.minimum(dist,od);K=torch.cat([Prior.Old.basis(centers[k:k+16],centers,radii) for k in range(0,len(centers),16)]);assert torch.equal(K,torch.eye(len(K),dtype=K.dtype))
  for k in range(0,len(points),128):assert float(Prior.Old.basis(points[k:k+128],centers,radii).max())==0.
  original=q0.sequence(x);weights=torch.tensor(a['teacher_actions'],dtype=torch.float64)-original
  # K is exactly identity: direct readout assignment is its unique linear solve.
  q=extend(q0,p,len(centers));q.layers[-1].centers.copy_(centers);q.layers[-1].radii.copy_(radii);q.layers[-1].weights.copy_(weights)
  assert np.array_equal(q.sequence(x).clamp(-1,1).float().numpy(),a['actions'])
  for row in p['protected_rows']:
   z=torch.tensor(G.E.N.S.arrays(row['path'])['states']);assert torch.equal(q.sequence(z),q0.sequence(z))
  for name,value in q0.state_dict().items():assert torch.equal(q.state_dict()[name],value)
  q.sample=False;q.reset_noise();h=None;previous=None
  for i,z in enumerate(x):
   u,loc,h,_=q.step(z[None],h);assert np.array_equal(u[0].numpy(),a['actions'][i]);assert np.array_equal(np.zeros(64) if i==0 else previous,a['hidden_before'][i]);previous=h[0,0].numpy().copy()
  q.reset_noise()
 return q,dict(closed_form_identity_solves=1,new_Adam=0,new_RL=0,centers=len(centers),minimum_protected_separation=float(dist.min()),minimum_other_center_separation=float(od.min()),minimum_radius=float(radii.min()),K_exact_identity=True,all_protected_outputs_exact=True,all_previous_model_state_exact=True,teacher_step_actions_and_hidden_exact=True,finite_memory_NOT_generalization=True)
def export(q,p,path):
 state=dict(actor=copy.deepcopy(q.state_dict()));b=io.BytesIO();torch.save(dict(architecture=ARCH,parent=p['initial'],layer=copy.deepcopy(q.layers[-1].state_dict()),policy_hash=q.phash(),joint_hash=digest(state),optimizer=None),b);blob=gzip.compress(b.getvalue(),mtime=0);assert len(blob)<=p['model_bytes']
 with Path(path).open('xb') as f:f.write(blob);f.flush();__import__('os').fsync(f.fileno())
 return dict(path=str(path),sha256=sha(path),bytes=len(blob),policy_hash=q.phash(),joint_hash=digest(state),parent=p['initial'])
