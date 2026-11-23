"""Fixed 8D projection and one global support; complete T694 stays frozen."""
from pathlib import Path
import sys, math
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import t699_actor as Old
torch,np,digest=Old.torch,Old.np,Old.digest
P=Old.Parent;BASE_FROZEN=Old.frozen
ARCH='T702_frozen_T694_fixed8D_shared_radius_float64'
SEED=202611230702

def project(features,matrix):
    # Exactly the same 184-element reduction for each coordinate in batch1,
    # construction and independent reconstruction. Never switch to GEMM.
    return torch.stack([torch.stack([(row*matrix[:,j]).sum() for j in range(8)])
                        for row in features.reshape(-1,184)])

class Actor(P.Actor):
    def __init__(self,p,old_count,count):
        super().__init__(p,old_count)
        self.register_buffer('projection',torch.zeros(184,8,dtype=torch.float64))
        self.register_buffer('projected_centers',torch.zeros(count,8,dtype=torch.float64))
        self.register_buffer('shared_radius',torch.ones(1,dtype=torch.float64))
        self.projected_weights=torch.nn.Parameter(torch.zeros(count,2,dtype=torch.float64))
        self.requires_grad_(False)
    def location(self,h,n):
        base=super().location(h,n);f=torch.cat([h,n],-1).reshape(-1,184)
        z=project(f,self.projection);phi=P.basis(z,self.projected_centers,self.shared_radius)
        correction=(phi@self.projected_weights).reshape(base.shape)
        active=(phi!=0).any(-1).reshape(base.shape[:-1])[...,None]
        return torch.where(active,base+correction,base)
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def frozen(p,path):
    saved=Old.Existing.Base.P.checkpoint(path)
    if saved.get('architecture')!=ARCH:return BASE_FROZEN(p,path)
    state=saved['joint_state']['actor'];q=Actor(p,len(state['compat_radii']),len(state['projected_centers']))
    q.load_state_dict(state);assert q.phash()==saved['policy_hash'];return q.eval()

def export(q,p,path):return Old.export(q,p,path,ARCH)

def construct(p,data,guard,event):
    parent=P.frozen(p,p['initial']['path'])
    generator=torch.Generator(device='cpu').manual_seed(SEED)
    matrix=torch.randn((184,8),generator=generator,dtype=torch.float64)/math.sqrt(8)
    chunks=[]
    for start in range(0,len(data['x']),128):
        guard();chunks.append(project(data['x'][start:start+128],matrix))
    z=torch.cat(chunks);old=z[data['groups']<403];centers=z[data['groups']>=403]
    assert len(old)==81956 and len(centers)==1737
    minimum=torch.tensor(float('inf'),dtype=torch.float64)
    for start in range(0,len(old),128):
        guard();distance=((centers[:,None]-old[start:start+128])**2).sum(-1)
        minimum=torch.minimum(minimum,distance.min())
    for start in range(0,len(centers),128):
        guard();distance=((centers[:,None]-centers[start:start+128])**2).sum(-1)
        count=min(128,len(centers)-start)
        distance[torch.arange(start,start+count),torch.arange(count)]=float('inf')
        minimum=torch.minimum(minimum,distance.min())
    radius=.49*minimum.sqrt()
    assert torch.isfinite(radius) and radius>1e-10,'Fixed shared radius infeasible; no seed/dimension/radius rescue'
    for start in range(0,len(old),128):
        guard();assert not P.basis(old[start:start+128],centers,radius.reshape(1)).any(),'Protected basis leak'
    for start in range(0,len(centers),128):
        guard();K=P.basis(centers[start:start+128],centers,radius.reshape(1))
        a=torch.arange(len(K));b=torch.arange(start,start+len(K))
        assert torch.equal(K[a,b],torch.ones(len(K),dtype=torch.float64))
        K[a,b]=0;assert not K.any(),'Source matrix is not identity'
    target=data['commands'][data['groups']>=403].float().double()
    coefficients=target-data['mu'][data['groups']>=403];assert torch.isfinite(coefficients).all()
    q=Actor(p,len(parent.compat_radii),1737)
    missing,extra=q.load_state_dict(parent.state_dict(),strict=False)
    assert set(missing)=={'projection','projected_centers','shared_radius','projected_weights'} and not extra
    q.projection.copy_(matrix);q.projected_centers.copy_(centers);q.shared_radius.copy_(radius.reshape(1))
    event('BEFORE',0,'closed_form');q.projected_weights.copy_(coefficients);event('AFTER',0,'closed_form')
    assert all(torch.equal(q.state_dict()[k],v) for k,v in parent.state_dict().items())
    added=sum(v.numel()*v.element_size() for k,v in q.state_dict().items() if k not in parent.state_dict())
    assert added==150744 and added<=161584
    return q,dict(seed=SEED,dimension=8,centers=1737,protected_decisions=81956,
        projection_hash=digest(matrix),projected_data_hash=digest(z),shared_radius=float(radius),minimum_squared_separation=float(minimum),
        old_basis_exact_zero=True,source_K_exact_identity=True,state_hash=digest(q.state_dict()),added_tensor_bytes=added,
        one_seed_one_dimension_one_radius_rule=True,new_Adam=0,new_RL=0)
