"""Euclidean actual-displacement projection onto at most eight halfspaces.

Pure NumPy active-set KKT enumeration. No optimization package, simulator or
learned state. Numeric feasibility is an empirical linear condition only.
"""
import numpy as np

def project(delta,gradients):
    delta=np.asarray(delta,np.float64);raw=np.asarray(gradients,np.float64)
    assert delta.ndim==1 and raw.ndim==2 and raw.shape[1]==len(delta) and len(raw)<=8
    assert np.isfinite(delta).all() and np.isfinite(raw).all()
    norms=np.linalg.norm(raw,axis=1);kept=np.flatnonzero(norms>0)
    norm=float(np.linalg.norm(delta));scale=max(norm,1e-16)
    if not len(kept) or norm==0:
        return delta.copy(),dict(active=[],normalized_before=[],normalized_after=[],proposal_norm=norm,projected_norm=norm,removed_fraction=0.,reference_norms=norms.tolist(),zero_rows=int((norms==0).sum()))
    g=raw[kept]/norms[kept,None];gram=g@g.T;u=g@delta
    tol=1e-12*scale;lam=np.zeros(len(g));active=[]
    if np.max(u)>tol:
        found=False
        for mask in range(1,1<<len(g)):
            idx=[j for j in range(len(g)) if mask&(1<<j)]
            z=np.linalg.lstsq(gram[np.ix_(idx,idx)],u[idx],rcond=1e-12)[0]
            if np.min(z)<-tol:continue
            z=np.maximum(z,0.);candidate=np.zeros(len(g));candidate[idx]=z
            residual=u-gram@candidate
            if np.max(np.abs(residual[idx]))>tol or np.max(residual)>tol:continue
            lam=candidate;active=[int(kept[j]) for j in idx];found=True;break
        assert found,'No numerically valid KKT active set; stop rather than relax tolerance'
    out=delta-g.T@lam;after=g@out
    assert float(np.max(after))<=1e-10*scale,('Projected displacement violated linear memory constraints',after)
    assert np.linalg.norm(out)<=norm+1e-10*scale
    return out,dict(active=active,normalized_before=u.tolist(),normalized_after=after.tolist(),proposal_norm=norm,
        projected_norm=float(np.linalg.norm(out)),removed_fraction=float(np.linalg.norm(delta-out)/scale),reference_norms=norms.tolist(),zero_rows=int((norms==0).sum()),dual_multipliers=lam.tolist(),kept_rows=kept.tolist(),kkt_complementarity=float(np.max(np.abs(lam*after))),kkt_violation=max(0.,float(after.max())))
