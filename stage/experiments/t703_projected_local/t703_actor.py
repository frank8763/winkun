"""T702 projection/centers/coefficients frozen; one nearest-protected radius each."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t702_projected_global'))
import t702_actor as Previous
torch,np,digest=Previous.torch,Previous.np,Previous.digest
P,Old,project=Previous.P,Previous.Old,Previous.project
SEED=Previous.SEED
ARCH='T703_frozen_T702_projection_centers_weights_local_fp32_radii'

class Actor(P.Actor):
    def __init__(self,p,old_count,count):
        super().__init__(p,old_count)
        self.register_buffer('projection',torch.zeros(184,8,dtype=torch.float64))
        self.register_buffer('projected_centers',torch.zeros(count,8,dtype=torch.float64))
        self.register_buffer('projected_radii',torch.ones(count,dtype=torch.float32))
        self.projected_weights=torch.nn.Parameter(torch.zeros(count,2,dtype=torch.float64))
        self.requires_grad_(False)
    def location(self,h,n):
        base=super().location(h,n);features=torch.cat([h,n],-1).reshape(-1,184)
        phi=P.basis(project(features,self.projection),self.projected_centers,self.projected_radii.double())
        correction=(phi@self.projected_weights).reshape(base.shape)
        active=(phi!=0).any(-1).reshape(base.shape[:-1])[...,None]
        return torch.where(active,base+correction,base)
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def frozen(p,path):
    saved=Old.Existing.Base.P.checkpoint(path)
    if saved.get('architecture')!=ARCH:return Previous.frozen(p,path)
    state=saved['joint_state']['actor'];q=Actor(p,len(state['compat_radii']),len(state['projected_centers']))
    q.load_state_dict(state);assert q.phash()==saved['policy_hash'];return q.eval()

def export(q,p,path):return Old.export(q,p,path,ARCH)

def construct(p,data,guard,event):
    model=p['projection_reference'];assert Old.Existing.sha(model['path'])==model['sha256']
    reference=Previous.frozen(p,model['path']);matrix=reference.projection.detach()
    chunks=[]
    for start in range(0,len(data['x']),128):
        guard();chunks.append(project(data['x'][start:start+128],matrix))
    z=torch.cat(chunks);old=z[data['groups']<403];centers=z[data['groups']>=403]
    assert len(old)==81956 and len(centers)==1737
    assert torch.equal(centers,reference.projected_centers),'Original source projections changed'
    nearest=torch.full((1737,),float('inf'),dtype=torch.float64)
    for start in range(0,len(old),128):
        guard();distance=((centers[:,None]-old[start:start+128])**2).sum(-1)
        nearest=torch.minimum(nearest,distance.min(-1).values)
    for start in range(0,len(centers),128):
        guard();distance=((centers[:,None]-centers[start:start+128])**2).sum(-1)
        count=min(128,len(centers)-start)
        distance[torch.arange(start,start+count),torch.arange(count)]=float('inf')
        nearest=torch.minimum(nearest,distance.min(-1).values)
    exact=.49*nearest.sqrt();stored=exact.float();radii=stored.double()
    assert torch.isfinite(radii).all() and (radii>1e-10).all() and (radii<nearest.sqrt()).all(),'Fixed radius quantization infeasible'
    assert float(exact.min())==float(reference.shared_radius[0])
    for start in range(0,len(old),128):
        guard();assert not P.basis(old[start:start+128],centers,radii).any(),'Protected basis leak'
    for start in range(0,len(centers),128):
        guard();K=P.basis(centers[start:start+128],centers,radii)
        a=torch.arange(len(K));b=torch.arange(start,start+len(K))
        assert torch.equal(K[a,b],torch.ones(len(K),dtype=torch.float64));K[a,b]=0
        assert not K.any(),'Source matrix not identity'
    q=Actor(p,len(reference.compat_radii),1737)
    state={k:v for k,v in reference.state_dict().items() if k!='shared_radius'}
    missing,extra=q.load_state_dict(state,strict=False);assert missing==['projected_radii'] and not extra
    event('BEFORE',0,'closed_form_radius_assignment');q.projected_radii.copy_(stored);event('AFTER',0,'closed_form_radius_assignment')
    assert all(torch.equal(q.state_dict()[k],v) for k,v in state.items())
    parent=P.frozen(p,p['initial']['path'])
    added=sum(v.numel()*v.element_size() for k,v in q.state_dict().items() if k not in parent.state_dict())
    assert added==157684 and added<161584
    ratio=radii/reference.shared_radius[0]
    return q,dict(projection_reference_sha256=model['sha256'],projection_hash=digest(matrix),projected_data_hash=digest(z),
        centers_hash=digest(centers),unchanged_source_coefficients_hash=digest(q.projected_weights),
        radius_dtype='float32_storage_float64_evaluation',radii_hash=digest(stored),
        minimum_radius=float(radii.min()),median_radius=float(radii.median()),maximum_radius=float(radii.max()),
        expansion_min=float(ratio.min()),expansion_median=float(ratio.median()),expansion_max=float(ratio.max()),
        old_basis_exact_zero=True,source_K_exact_identity=True,protected_decisions=81956,centers=1737,
        state_hash=digest(q.state_dict()),added_tensor_bytes=added,no_seed_dimension_radius_search=True,new_Adam=0,new_RL=0)
