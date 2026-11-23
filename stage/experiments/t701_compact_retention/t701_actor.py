"""Center-only storage compression; original float64 forward is preserved."""
import copy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t699_matched_value'))
import t699_actor as Old
torch,np,digest=Old.torch,Old.np,Old.digest
P=Old.Parent
ARCH='T701_frozen_T694_fp32_centers_float64_cardinal_evaluation'
ORIGINAL_FROZEN=Old.frozen

class Actor(P.Actor):
    def __init__(self,p,old_count,count):
        super().__init__(p,old_count)
        self.register_buffer('new_centers',torch.zeros(count,184,dtype=torch.float32))
        self.register_buffer('new_radii',torch.ones(count,dtype=torch.float64))
        self.new_weights=torch.nn.Parameter(torch.zeros(count,2,dtype=torch.float64))
        self.requires_grad_(False)
    def location(self,h,n):
        original=super().location(h,n);f=torch.cat([h,n],-1).reshape(-1,184)
        values=[];active=[]
        # Decoding is deliberately ephemeral. Count this workspace in measured
        # peak memory and conversion time; don't claim arithmetic acceleration.
        centers=self.new_centers.double()
        for start in range(0,len(f),32):
            phi=P.basis(f[start:start+32],centers,self.new_radii)
            values.append(phi@self.new_weights);active.append((phi!=0).any(-1))
        correction=torch.cat(values).reshape(original.shape)
        mask=torch.cat(active).reshape(original.shape[:-1])[...,None]
        return torch.where(mask,original+correction,original)
    def phash(self):return digest(dict(architecture=ARCH,state=self.state_dict(),sigma=self.p['raw_sigma']))

def frozen(p,path):
    saved=Old.Existing.Base.P.checkpoint(path)
    if saved.get('architecture')!=ARCH:return ORIGINAL_FROZEN(p,path)
    state=saved['joint_state']['actor'];q=Actor(p,len(state['compat_radii']),len(state['new_radii']))
    q.load_state_dict(state);assert q.phash()==saved['policy_hash']
    return q.eval()

def construct(p,d,guard,event):
    parent=P.frozen(p,p['initial']['path']);source=d['x'][d['groups']>=403]
    protected=d['x'][d['groups']<403];centers=source.float().double()
    assert len(source)==1737 and len(protected)==81956
    nearest=torch.full((len(source),),float('inf'),dtype=torch.float64)
    for start in range(0,len(protected),32):
        guard();dist=((centers[:,None]-protected[start:start+32])**2).sum(-1)
        nearest=torch.minimum(nearest,dist.min(-1).values)
    other=torch.full_like(nearest,float('inf'))
    for start in range(0,len(source),32):
        guard();dist=((centers[:,None]-source[start:start+32])**2).sum(-1)
        dist[torch.arange(start,min(start+32,len(source))),torch.arange(min(32,len(source)-start))]=float('inf')
        other=torch.minimum(other,dist.min(-1).values)
    radii=.49*torch.minimum(nearest,other).sqrt()
    squared_error=((source-centers)**2).sum(-1);error=squared_error.sqrt()
    assert torch.isfinite(radii).all() and (radii>1e-10).all() and (error<radii).all(),'Fixed fp32 diagonal geometry infeasible'
    diagonal=torch.relu(1-squared_error/radii.square()).square()
    # Labels retain their original bytes; canonical executed float32 values
    # lifted exactly to float64 define this reconstruction's command target.
    target=d['commands'][d['groups']>=403].float().double()
    base=d['mu'][d['groups']>=403]
    coefficients=(target-base)/diagonal[:,None]
    assert torch.isfinite(coefficients).all()
    for start in range(0,len(protected),32):
        guard();assert not P.basis(protected[start:start+32],centers,radii).any(),'Protected basis leak'
    for start in range(0,len(source),32):
        guard();K=P.basis(source[start:start+32],centers,radii)
        diag=K[torch.arange(len(K)),torch.arange(start,start+len(K))].clone()
        assert torch.equal(diag,diagonal[start:start+len(K)])
        K[torch.arange(len(K)),torch.arange(start,start+len(K))]=0
        assert not K.any(),'Off-diagonal original-source support leak'
    q=Actor(p,len(parent.compat_radii),len(source))
    missing,extra=q.load_state_dict(parent.state_dict(),strict=False)
    assert set(missing)=={'new_centers','new_radii','new_weights'} and not extra
    q.new_centers.copy_(source.float());q.new_radii.copy_(radii)
    event('BEFORE',0,'closed_form');q.new_weights.copy_(coefficients);event('AFTER',0,'closed_form')
    assert all(torch.equal(q.state_dict()[k],v) for k,v in parent.state_dict().items())
    return q,dict(centers=1737,protected_decisions=81956,center_storage='float32',computation='float64',
        protected_basis_exact_zero=True,original_source_K_offdiagonal_exact_zero=True,
        diagonal_hash=digest(diagonal),minimum_diagonal=float(diagonal.min()),
        maximum_quantization_error=float(error.max()),minimum_radius=float(radii.min()),
        state_hash=digest(q.state_dict()),no_center_selection_or_radius_rescue=True)

def export(q,p,path):return Old.export(q,p,path,ARCH if isinstance(q,Actor) else Old.ARCH)
