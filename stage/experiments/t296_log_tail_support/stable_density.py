"""Log-domain audit of the SAME bounded Gaussian-copula AR density.

No policy, sampling, sigma, rho, clipping or objective changes. No Actor updates.
SciPy's documented inverse-log-CDF approach motivates keeping tails in log space;
we use a checked torch Newton solve and independent 90-digit mpmath reference.
"""
import math
import numpy as np
import torch
SIGMA=(.035,.03);RHO=.98;LOG2PI=math.log(2*math.pi)

def loginterval(lo,hi):
    assert bool((lo<hi).all())
    flip=lo>=0
    a=torch.where(flip,-lo,hi);b=torch.where(flip,-hi,lo)
    la=torch.special.log_ndtr(a);lb=torch.special.log_ndtr(b)
    out=la+torch.log(-torch.expm1(lb-la));assert bool(torch.isfinite(out).all());return out

def invlogcdf(logp):
    assert bool((logp<=math.log(.5)+1e-10).all()) and bool(torch.isfinite(logp).all())
    # Initial guess only; neither the target log probability nor weights clipped.
    z=-torch.sqrt(-2*logp)
    for _ in range(12):
        lc=torch.special.log_ndtr(z);derivative=torch.exp(-.5*z*z-.5*LOG2PI-lc);z=z-(lc-logp)/derivative
    assert bool(torch.isfinite(z).all())
    assert float((torch.special.log_ndtr(z)-logp).detach().abs().max())<1e-7
    return z

def terms(mean,action):
    m=mean.double();a=action.double();s=torch.tensor(SIGMA,dtype=torch.float64)
    assert bool(torch.isfinite(m).all()) and bool(torch.isfinite(a).all()) and bool((a.abs()<1).all())
    lo=(-1-m)/s;hi=(1-m)/s;t=(a-m)/s;mass=loginterval(lo,hi)
    lu=loginterval(lo,t)-mass;lv=loginterval(t,hi)-mass
    assert float(torch.logaddexp(lu,lv).detach().abs().max())<1e-7
    lower=lu<=lv;small=torch.where(lower,lu,lv);negative=invlogcdf(small);z=torch.where(lower,negative,-negative)
    marginal=-.5*t*t-s.log()-.5*LOG2PI-mass
    return z,marginal

def logp(mean,action,previous_mean,previous_action,reset):
    z,marginal=terms(mean,action);prev,_=terms(previous_mean,previous_action)
    rr=torch.where(reset.bool(),torch.zeros_like(z[...,0]),torch.full_like(z[...,0],RHO))[...,None];var=1-rr*rr
    value=(-.5*(z-rr*prev).square()/var-.5*var.log()+.5*z.square()+marginal).sum(-1)
    assert bool(torch.isfinite(value).all());return value

def reference_contract():
    import mpmath as mp
    mp.mp.dps=90
    means=np.array([[-2.,2.],[-1.2,1.1],[-.4,.7],[0.,0.],[.7,-.8],[1.1,-1.2],[2.,-2.]])
    actions=np.array([[.999999,-.999999],[.999,-.999],[.98,-.97],[.01,-.01],[-.9,.9],[-.99,.99],[-.999999,.999999]])
    z,m=terms(torch.tensor(means),torch.tensor(actions));reference_z=[];reference_m=[]
    def interval(lo,hi):
        sqrt2=mp.sqrt(2)
        if lo>=0:return (mp.erfc(lo/sqrt2)-mp.erfc(hi/sqrt2))/2
        return (mp.erfc(-hi/sqrt2)-mp.erfc(-lo/sqrt2))/2
    for i in range(len(means)):
        zz=[];mm=[]
        for j in range(2):
            mean=mp.mpf(str(means[i,j]));action=mp.mpf(str(actions[i,j]));sigma=mp.mpf(str(SIGMA[j]));lo=(-1-mean)/sigma;hi=(1-mean)/sigma;t=(action-mean)/sigma
            mass=interval(lo,hi);lu=mp.log(interval(lo,t)/mass);lv=mp.log(interval(t,hi)/mass);lower=lu<=lv;small=lu if lower else lv
            guess=-abs(float(z[i,j]));root=mp.findroot(lambda x:mp.log(mp.erfc(-x/mp.sqrt(2))/2)-small,(guess-.01,guess+.01),tol=mp.mpf('1e-65'),maxsteps=80)
            assert abs(mp.log(mp.erfc(-root/mp.sqrt(2))/2)-small)<mp.mpf('1e-60')
            zz.append(float(root if lower else -root));mm.append(float(-t*t/2-mp.log(sigma)-mp.log(2*mp.pi)/2-mp.log(mass)))
        reference_z.append(zz);reference_m.append(mm)
    ze=float(np.max(np.abs(z.numpy()-reference_z)));me=float(np.max(np.abs(m.numpy()-reference_m)));assert ze<1e-8 and me<1e-8
    # Check derivatives through the log-CDF inverse and previous conditional mean.
    x=torch.tensor([[1.1,-.3]],dtype=torch.float64,requires_grad=True);px=torch.tensor([[1.05,-.25]],dtype=torch.float64,requires_grad=True);a=torch.tensor([[.95,.65]],dtype=torch.float64);pa=torch.tensor([[.94,.6]],dtype=torch.float64);reset=torch.tensor([False])
    value=logp(x,a,px,pa,reset);gx,gp=torch.autograd.grad(value.sum(),[x,px]);errors=[];eps=1e-6
    for name,v,g in [('current',x,gx),('previous',px,gp)]:
        for j in range(2):
            plus=v.detach().clone();minus=plus.clone();plus[0,j]+=eps;minus[0,j]-=eps
            vp=logp(plus if name=='current' else x.detach(),a,plus if name=='previous' else px.detach(),pa,reset)
            vm=logp(minus if name=='current' else x.detach(),a,minus if name=='previous' else px.detach(),pa,reset)
            errors.append(abs(float((vp-vm)/(2*eps))-float(g[0,j])))
    assert max(errors)<1e-3
    return dict(state='PASS',mpmath_dps=90,extreme_scalar_cases=14,max_z_error=ze,max_log_marginal_error=me,max_AR_current_previous_mean_gradient_fd_error=max(errors),native_steps=0,optimizer_calls=0,Actor_updates=0,no_density_probability_or_weight_clipping=True)
