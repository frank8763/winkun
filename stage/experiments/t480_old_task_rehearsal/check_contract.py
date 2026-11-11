"""Exact finite-sequence V-trace checks; zero real optimizer/native calls."""
import json
import torch
import t480_learning as L


def main():
    torch.set_num_threads(1)
    r=torch.tensor([.2,-.1,1.],dtype=torch.float64)
    d=torch.tensor([.9,.8,0.],dtype=torch.float64)
    v=torch.tensor([.3,.5,-.2,0.],dtype=torch.float64,requires_grad=True)
    log=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    target,adv,w=L.vtrace(r,d,v,log)
    expected=torch.tensor([.2+.9*(-.1+.8),-.1+.8,1.],dtype=torch.float64)
    torch.testing.assert_close(target,expected,rtol=0,atol=1e-15)
    torch.testing.assert_close(adv,expected-v[:-1].detach(),rtol=0,atol=1e-15)
    assert not target.requires_grad and not adv.requires_grad and not w.requires_grad
    assert torch.equal(w,torch.ones(3,dtype=torch.float64))
    # Independent recursive enumeration for nontrivial off-policy ratios.
    ratios=[.5,2.,.25]
    log=torch.tensor(ratios,dtype=torch.float64).log().requires_grad_()
    target,adv,w=L.vtrace(r,d,v,log)
    baseline=v.detach().tolist();rv=r.tolist();discount=d.tolist();weights=[min(1.,x) for x in ratios]
    vs=baseline.copy()
    for i in reversed(range(3)):
        vs[i]=baseline[i]+weights[i]*(rv[i]+discount[i]*baseline[i+1]-baseline[i])+discount[i]*weights[i]*(vs[i+1]-baseline[i+1])
    pg=[weights[i]*(rv[i]+discount[i]*vs[i+1]-baseline[i]) for i in range(3)]
    torch.testing.assert_close(target,torch.tensor(vs[:3],dtype=torch.float64),rtol=0,atol=1e-15)
    torch.testing.assert_close(adv,torch.tensor(pg,dtype=torch.float64),rtol=0,atol=1e-15)
    loss=-(log*adv).mean();loss.backward()
    assert v.grad is None and log.grad is not None
    try:L.vtrace(r,d,torch.tensor([.3,.5,-.2,.1],dtype=torch.float64),log)
    except AssertionError:pass
    else:raise AssertionError('Nonzero finite terminal bootstrap admitted')
    print(json.dumps(dict(state='PASS_ZERO_NATIVE_ZERO_REAL_ADAM',native=0,optimizer=0,
        synthetic_backward=1,onpolicy_exact_MC=True,clipped_offpolicy_recursion=True,
        detached_targets_and_policy_advantages=True,terminal_bootstrap_rejection=True,
        actual_behavior_full_history_preflight_NOT_yet_executed=True)))


if __name__=='__main__':main()
