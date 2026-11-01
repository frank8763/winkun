"""Fixed-rank-rule projection of an existing displacement; not an optimizer."""
import numpy as np
import torch


def project_displacement(jacobian, delta, relative_cutoff=1e-8):
    j=jacobian.double();d=delta.double()
    sizes=torch.linalg.vector_norm(j,dim=1)
    valid=sizes>1e-12;j=j[valid]/sizes[valid,None]
    gram=j@j.T;values,vectors=torch.linalg.eigh(gram)
    cutoff=max(float(values.max())*relative_cutoff,1e-12)
    keep=values>cutoff;v=vectors[:,keep]
    coeff=v@((v.T@(j@d))/values[keep])
    projected=d-j.T@coeff
    assert torch.isfinite(projected).all()
    assert torch.linalg.vector_norm(projected)<=torch.linalg.vector_norm(d)*(1+1e-8)
    return projected,dict(rows=int(jacobian.shape[0]),valid_rows=int(valid.sum()),rank=int(keep.sum()),
        eigenvalues=values.tolist(),eigenvalue_cutoff=cutoff,
        relative_remaining_displacement=float(torch.linalg.vector_norm(projected)/torch.linalg.vector_norm(d)),
        normalized_linear_action_change_before=float(torch.linalg.vector_norm(j@d)),
        normalized_linear_action_change_after=float(torch.linalg.vector_norm(j@projected)))


def flat_params(model):return torch.cat([p.detach().reshape(-1).double() for p in model.parameters() if p.requires_grad])


def install(model,vector):
    off=0
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(vector[off:off+p.numel()].reshape_as(p).to(p.dtype));off+=p.numel()
    assert off==vector.numel()


def jacobian_rows(model,states):
    params=[p for p in model.parameters() if p.requires_grad];rows=[]
    for state in states:
        action=model.step(torch.tensor(state[None]))[0][0]
        for column in range(2):
            gradients=torch.autograd.grad(action[column],params,retain_graph=column==0,allow_unused=True)
            rows.append(torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,gradients)]).detach())
    return torch.stack(rows)


def synthetic_contract():
    j=torch.tensor([[1.,0.,0.],[2.,0.,0.],[0.,1.,0.]],dtype=torch.float64)
    d=torch.tensor([2.,3.,4.],dtype=torch.float64);p,info=project_displacement(j,d)
    assert info['rank']==2 and torch.allclose(p,torch.tensor([0.,0.,4.],dtype=torch.float64),atol=1e-12,rtol=0)
    again,_=project_displacement(j,p);assert torch.allclose(again,p,atol=1e-12,rtol=0)
    return dict(rank_deficient_analytic_solution=True,idempotence=True,optimizer_calls=0,native_steps=0)
