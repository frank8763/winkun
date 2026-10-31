from pathlib import Path
import sys,json,time,hashlib,fcntl
import numpy as np
import torch
from copula import *
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t238_ar_density_recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'summary.json').exists();plan=json.loads((HERE/'protocol.json').read_text());start=time.monotonic();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for f,h in plan['input_hashes'].items():assert sha(f)==h
    m=torch.tensor([[.3,.95]],dtype=torch.float64);pm=torch.tensor([[.25,.9]],dtype=torch.float64);pa=sample(pm,[[.4,-.7]]).double();a=sample(m,[[.6,-.5]]).double();reset=torch.zeros(1,dtype=torch.bool)
    # Integrate in a finite conditional-latent interval. da/dz comes from
    # automatic differentiation of the continuous inverse-CDF sampler, while
    # density comes from the separate forward-CDF implementation.
    x,w=np.polynomial.legendre.leggauss(plan['quadrature_nodes']);xx=torch.tensor(x);ww=torch.tensor(w);norm=[]
    zp=terms(pm,pa)[0];scale=7*math.sqrt(1-RHO**2)
    for j in range(2):
        zz=zp.expand(len(x),-1).clone()*RHO;zz[:,j]+=scale*xx;zz.requires_grad_(True)
        aa=sample(m.expand(len(x),-1),zz,native=False);jac=torch.autograd.grad(aa[:,j].sum(),zz)[0][:,j]
        z,mg=terms(m.expand(len(x),-1),aa);v=1-RHO**2
        lp=-.5*(z[:,j]-RHO*zp[0,j]).square()/v-.5*math.log(v)-.5*LOG2PI+mg[:,j]+.5*z[:,j].square()+.5*LOG2PI
        norm.append(float((lp.exp()*jac*ww*scale).sum().detach()))
    marginal=terms(m,a)[1].sum(-1);identity=max(float((logp(m,a,pm,pa,reset,rho=0)-marginal).abs().max()),float((logp(m,a,pm,pa,~reset)-marginal).abs().max()))
    p=torch.cat([m.flatten(),pm.flatten()]).requires_grad_(True)
    fn=lambda q:logp(q[:2][None],a,q[2:][None],pa,reset).sum()
    gradient=torch.autograd.grad(fn(p),p)[0];fd=[]
    for j in range(4):
        d=torch.zeros(4,dtype=torch.float64);d[j]=1e-6;fd.append(float((fn(p.detach()+d)-fn(p.detach()-d))/(2e-6)))
    fd=torch.tensor(fd,dtype=torch.float64);fd_error=float(((gradient-fd).abs()/(1+fd.abs())).max())
    rng=np.random.default_rng(plan['seed']);previous=terms(pm,pa)[0].numpy();latent=RHO*previous+math.sqrt(1-RHO**2)*rng.normal(size=(plan['draws'],2));means=m.expand(plan['draws'],-1).clone().requires_grad_(True);prevmeans=pm.expand(plan['draws'],-1).clone().requires_grad_(True);acts=sample(means.detach(),latent);prioracts=pa.expand(plan['draws'],-1);rr=torch.zeros(plan['draws'],dtype=torch.bool)
    lp=logp(means,acts,prevmeans,prioracts,rr);scores=torch.cat(torch.autograd.grad(lp.sum(),(means,prevmeans)),dim=1);se=scores.std(0,unbiased=True)/math.sqrt(plan['draws']);zscore=scores.mean(0).abs()/se
    recovered=terms(means.detach(),acts)[0];roundtrip=float((recovered-torch.tensor(latent)).abs().max());normalization=max(abs(v-1) for v in norm)
    # Actual large positive and negative means: finite tails rather than zero mass.
    tailm=torch.tensor([[1.3,-1.3],[1.75,-1.75]],dtype=torch.float64);taila=sample(tailm,torch.zeros_like(tailm));tailz,tailmg=terms(tailm,taila);assert torch.isfinite(tailmg).all() and torch.isfinite(tailz).all()
    gates=plan['gates'];passed=normalization<=gates['normalization_abs_max'] and fd_error<=gates['finite_difference_relative_max'] and identity<=gates['reset_marginal_max'] and float(zscore.max())<=gates['score_mean_max_standard_errors'] and roundtrip<=gates['roundtrip_latent_max'] and time.monotonic()-start<plan['wall_seconds']
    r=dict(state='PREFLIGHT_PASS' if passed else 'PREFLIGHT_FAILED',conditional_normalization=norm,normalization_max_error=normalization,reset_or_rho0_marginal_error=identity,gradient=gradient.tolist(),finite_difference=fd.tolist(),finite_difference_max_relative_error=fd_error,score_mean=scores.mean(0).tolist(),score_mean_standard_errors=zscore.tolist(),roundtrip_latent_max_error=roundtrip,stable_extreme_tail_actions=taila.tolist(),previous_mean_gradient_checked=True,physics_steps=0,optimizer_calls=0,elapsed_s=time.monotonic()-start)
    for f,h in plan['input_hashes'].items():assert sha(f)==h
    (OUT/'summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
