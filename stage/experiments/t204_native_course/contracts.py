"""Zero-optimizer checks for the new score function and immutable backbone."""
import os
os.sched_setaffinity(0,{0})
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t204_native_course'
sys.path.insert(0,str(ROOT/'stage/experiments/t203_native_exploration'))
from runtime import *
def main():
    assert not (OUT/'contracts.json').exists();torch.set_num_threads(1);plan=json.loads((OUT/'protocol.json').read_text())
    p=Policy().eval();parent=torch.load(plan['checkpoint'],map_location='cpu',weights_only=False)['policy'];p.load_state_dict(parent);params=p.heads_only();assert sum(x.numel() for x in params)==2410
    with np.load(ROOT/'stage/runs/t196_left_exploration/retention_targets.npz') as z:x=torch.tensor(z['states']);target=torch.tensor(z['actions'])
    with torch.no_grad():error=float((p(x)-target).abs().max())
    assert error==0
    # Independently differentiate the normalized truncated-density formula.
    checks=[]
    for m in [-1.74,-1.2,0.,1.2,1.74]:
        mu=torch.tensor([[m,.2]],dtype=torch.float64,requires_grad=True);action=torch.tensor([[.7,-.1]],dtype=torch.float64);g=torch.autograd.grad(log_density(mu,action).sum(),mu)[0][0,0]
        sigma=SIGMA[0];lo=(-1-m)/sigma;hi=(1-m)/sigma;phi=lambda z:math.exp(-z*z/2)/math.sqrt(2*math.pi);cdf=lambda z:.5*math.erfc(-z/math.sqrt(2));expected=(.7-m)/sigma**2-(phi(lo)-phi(hi))/(sigma*(cdf(hi)-cdf(lo)))
        assert abs(float(g)-expected)<1e-9;checks.append({'mean':m,'gradient':float(g),'analytic':expected})
    # The group baseline is action-independent for its own rollout.
    r=np.asarray([1.,-.5,-1.]);a=r-(r.sum()-r)/2;assert np.allclose(a,[1.75,-.5,-1.25]) and abs(a.sum())<1e-12
    assert all(torch.equal(v,p.state_dict()[k]) for k,v in parent.items())
    raw=C.ParentPolicy.forward(p,x);mask=raw[:,0]>1.;assert bool(mask.any())
    actions=torch.zeros((int(mask.sum()),2));actions[:,0]=.5
    p.physical_logp(x[mask],actions).mean().backward()
    head_gradient_norm=float(torch.sqrt(sum(q.grad.square().sum() for q in params if q.grad is not None)))
    assert math.isfinite(head_gradient_norm) and head_gradient_norm>1e-8
    assert all(q.grad is None for q in p.parameters() if not q.requires_grad)
    assert all(torch.equal(v,p.state_dict()[k]) for k,v in parent.items())
    C.save(OUT/'contracts.json',{'state':'PASS','trainable_parameters':2410,'initial_retention_max_action_error':error,'analytic_score_checks':checks,'clipped_state_count':int(mask.sum()),'clipped_native_score_head_gradient_norm':head_gradient_norm,'optimizer_updates':0,'physics_steps':0,'scope':'Native truncated Gaussian is normalized/differentiable; exact initial deterministic function, nonzero head score gradients on clipped states, and frozen backbone. Does not establish navigation improvement or full training.'})
    print('T204 contracts PASS')
if __name__=='__main__':main()
