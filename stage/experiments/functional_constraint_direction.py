"""Small linearized-constraint projection primitive; not CPO/TRPO implementation."""
import itertools
import torch


def project_two_constraints(proposal, normals, margins):
    """Closest Euclidean delta s.t. A delta <= b, for at most two constraints.

    Exact active-set enumeration for the local linear problem. Nonlinear model
    constraints MUST be rechecked before accepting a parameter update.
    """
    proposal=proposal.double();normals=normals.double();margins=margins.double()
    if normals.ndim!=2 or len(normals)>2 or normals.shape[1]!=len(proposal):
        raise ValueError('At most two matching linearized constraints required')
    if not bool(torch.isfinite(proposal).all() and torch.isfinite(normals).all() and torch.isfinite(margins).all()):
        raise ValueError('Finite values required')
    norms=normals.norm(dim=1)
    if bool(((norms<1e-12)&(margins<0)).any()):
        raise ValueError('Unsatisfied constraint has no first-order direction')
    keep=norms>=1e-12
    A=normals[keep]/norms[keep,None];b=margins[keep]/norms[keep]
    choices=[]
    for count in range(len(A)+1):
        for subset in itertools.combinations(range(len(A)),count):
            if subset:
                active=A[list(subset)];rhs=active@proposal-b[list(subset)]
                multiplier=torch.linalg.pinv(active@active.T,rtol=1e-12)@rhs
                if float(multiplier.min()) < -1e-10:continue
                candidate=proposal-active.T@multiplier
                if float((active@candidate-b[list(subset)]).abs().max())>1e-9:continue
            else:
                candidate=proposal.clone();multiplier=proposal.new_empty(0)
            if len(A) and float((A@candidate-b).max())>1e-9:continue
            choices.append((float((candidate-proposal).square().sum()),candidate,subset,multiplier))
    if not choices:raise ValueError('No feasible active set')
    _,delta,active,multipliers=min(choices,key=lambda c:c[0])
    return delta.to(dtype=torch.float32),{'active_constraints':list(active),'multipliers':multipliers.tolist(),
        'linearized_max_violation':float((normals@delta-margins).max()) if len(normals) else 0.,
        'proposal_norm':float(proposal.norm()),'projected_norm':float(delta.norm())}
