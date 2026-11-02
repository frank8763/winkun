"""Counterfactual goal/history reconstruction and conditional-AR support audit.

No optimizer or Actor update. Fixed historical auxiliary goals, not an unbiased
estimate of the original six-metre objective. No importance-weight clipping.
"""
from pathlib import Path
import sys,math,itertools
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t287_time_aligned_credit'))
from t287_credit import L,S,C,R,np,torch,policy
sys.path.insert(0,str(ROOT/'stage/experiments/t290_exact_duplicate_store'))
from store import resolve

GOALS=np.array([48,49,50,51,52,72,73,84,85,96,97,108,109])
KEEP=np.setdiff1d(np.arange(120),GOALS)

def load(path):
    with np.load(resolve(path)) as z:return {k:z[k].copy() for k in z.files}

def fields(pose,goal):
    x,y,yaw=map(float,pose[:3]);dx,dy=map(float,np.asarray(goal)-[x,y])
    bx=math.cos(yaw)*dx+math.sin(yaw)*dy;by=-math.sin(yaw)*dx+math.cos(yaw)*dy
    d=math.hypot(dx,dy);angle=math.atan2(by,bx)
    return np.array([np.clip(d/10,0,2),math.sin(angle),math.cos(angle),np.clip(bx/10,-2,2),np.clip(by/10,-2,2)],dtype=np.float32),d,angle

def rebuild(x,pose,hp,goal,hg=None):
    out=x.copy();out[48:53]=fields(pose,goal)[0]
    for j,p in enumerate(hp):
        _,d,a=fields(p,goal if hg is None else hg[j]);out[72+12*j]=np.clip(d/10,0,2);out[73+12*j]=np.clip(a/math.pi,-1,1)
    assert np.array_equal(out[KEEP],x[KEEP]);return out

def density(q,states,actions):
    with torch.no_grad():means=torch.cat([q.step(torch.tensor(x[None]))[1] for x in states])
    acts=torch.tensor(actions);prev_m=torch.cat([means[:1],means[:-1]]);prev_a=torch.cat([acts[:1],acts[:-1]])
    reset=torch.zeros(len(states),dtype=torch.bool);reset[0]=True
    # terms() explicitly rejects non-invertible rounded actions; never clamp.
    return means.numpy(),L.AR.logp(means,acts,prev_m,prev_a,reset).numpy()

def weights(logweights):
    x=np.asarray(logweights,dtype=np.float64);assert np.isfinite(x).all()
    w=np.exp(x-x.max());w/=w.sum()
    return dict(log_weights=x.tolist(),log_weight_range=float(np.ptp(x)),normalized_weights=w.tolist(),ESS=float(1/np.sum(w*w)),max_weight=float(w.max()))

def enumeration():
    # Full enumeration with goal-independent binary transitions, a goal- and
    # previous-action-conditioned policy, and goal-dependent first-hit stopping.
    theta=torch.tensor(.23,dtype=torch.float64,requires_grad=True)
    direct=torch.zeros((),dtype=torch.float64);corrected=torch.zeros_like(direct);score=torch.zeros_like(direct)
    for acts in itertools.product([0,1],repeat=2):
        bprob=1.;ratio=torch.ones_like(theta);prob=torch.ones_like(theta);ret=0.;weighted=torch.zeros_like(theta);scoreterm=torch.zeros_like(theta);sumlog=torch.zeros_like(theta);x=0;prev=0;alive=True
        for k,a in enumerate(acts):
            b=1/(1+math.exp(-(-.17+.2*prev)));bprob*=b if a else 1-b
            pp=torch.sigmoid(theta+.35*2+.4*prev);pa=pp if a else 1-pp;prob=prob*pa
            ratio=ratio*pa/(b if a else 1-b);sumlog=sumlog+pa.log();x+=a
            if alive:
                reward=(1. if x>=1 else -.1)*(0.91**k)
                ret+=reward;weighted=weighted+ratio*reward;scoreterm=scoreterm+ratio.detach()*sumlog*reward
                if x>=1:alive=False
            prev=a
        direct=direct+prob*ret;corrected=corrected+bprob*weighted;score=score+bprob*scoreterm
    a=torch.autograd.grad(direct,theta,retain_graph=True)[0];b=torch.autograd.grad(corrected,theta,retain_graph=True)[0];c=torch.autograd.grad(score,theta)[0]
    assert abs(float(direct-corrected))<1e-13 and abs(float(a-b))<1e-13 and abs(float(a-c))<1e-13
    # Previous-mean derivative is essential for the real AR conditional density.
    m=torch.tensor([[.1,-.2]],dtype=torch.float64,requires_grad=True);pm=torch.tensor([[.05,-.1]],dtype=torch.float64,requires_grad=True)
    ac=torch.tensor([[.12,-.19]],dtype=torch.float64);pa=torch.tensor([[.06,-.11]],dtype=torch.float64)
    v=L.AR.logp(m,ac,pm,pa,torch.tensor([False]));g=torch.autograd.grad(v.sum(),pm)[0];eps=1e-6
    fd=[]
    for j in range(2):
        plus=pm.detach().clone();minus=plus.clone();plus[0,j]+=eps;minus[0,j]-=eps
        fd.append(float((L.AR.logp(m.detach(),ac,plus,pa,torch.tensor([False]))-L.AR.logp(m.detach(),ac,minus,pa,torch.tensor([False])))/(2*eps)))
    err=float(np.max(np.abs(g.numpy()[0]-fd)));assert err<1e-4 and float(g.abs().max())>1e-3
    assert weights([0.,0.,0.,0.])['ESS']==4
    return dict(state='PASS',native_steps=0,optimizer_calls=0,Actor_updates=0,finite_enumeration_value=float(direct.detach()),prefix_IS_value=float(corrected.detach()),exact_gradient=float(a),prefix_IS_gradient=float(c),AR_previous_mean_finite_difference_max_error=err,scope='Synthetic finite Markov-policy prefix-IS identity only; not a theorem for historical data-selected continuous goals or weighted finite-sample HPG.')
