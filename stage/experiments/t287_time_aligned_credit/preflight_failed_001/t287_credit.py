"""Physical-time aligned reward-to-go leave-one-out baseline.

Standard independent-trajectory baseline, not a new unbiasedness theorem or
universal variance reduction claim. Absorbing post-terminal values are zero.
"""
from pathlib import Path
import sys,itertools
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t267_causal_credit_course'))
from t267_learning import *
_causal_batch=L.training_batch


def physical_time_baselines(returns,counts,groups):
    starts=[np.r_[0,np.cumsum(c.astype(np.int64))[:-1]] for c in counts]
    tables=[dict(zip(t.tolist(),g.tolist())) for t,g in zip(starts,returns)]
    ends=[int(np.sum(c.astype(np.int64))) for c in counts];output=[]
    for j,(times,group) in enumerate(zip(starts,groups)):
        peers=[k for k,g in enumerate(groups) if k!=j and g==group];assert len(peers)==3
        baseline=[]
        for t in times:
            values=[]
            for k in peers:
                if int(t) in tables[k]:values.append(tables[k][int(t)])
                else:
                    assert int(t)>=ends[k],('No exact physical-time match before termination',t,ends[k])
                    values.append(0.)
            baseline.append(float(np.mean(values)))
        output.append(np.asarray(baseline,dtype=np.float64))
    return output


def training_batch(items,p):
    data=_causal_batch(items,p);a=[i['arrays'] for i in items]
    returns=[S.monte_carlo(z['rewards'],z['native_counts']) for z in a]
    groups=[i['row']['source_reset_id'] for i in items]
    baselines=physical_time_baselines(returns,[z['native_counts'] for z in a],groups)
    raw=torch.zeros_like(data['mask'],dtype=torch.float64)
    for j,(g,b) in enumerate(zip(returns,baselines)):raw[:len(g),j]=torch.tensor(g-b)
    scale=data['density']['original_estimator_RMS_scale']
    old=data['advantage'].clone();data['advantage']=raw[data['mask']]/scale
    assert torch.isfinite(data['advantage']).all()
    data['density'].update(causal_time_baseline=False,physical_time_aligned_RLOO=True,
        independent_peer_baseline=True,post_terminal_peer_return_zero=True,
        scale_unchanged_from_previous_estimator=True,
        prior_advantage_RMS=float(old.square().mean().sqrt()),new_advantage_RMS=float(data['advantage'].square().mean().sqrt()))
    return data


def enumeration_contract():
    # Complete two-step independent Bernoulli policy enumeration, no sampled seed.
    # Check the independent baseline contributes zero expected score at each time.
    theta=.3;prob=1/(1+np.exp(-theta));gamma=.91
    trajectories=list(itertools.product([0,1],repeat=2))
    def terms(trajectory):
        a,b=trajectory;r0=.2*a-.1;r1=1.3*b+.7*a*b
        return np.asarray([r0+gamma*r1,r1]),prob**(a+b)*(1-prob)**(2-a-b)
    direct=0.;baseline_effect=np.zeros(2);loo=0.
    for own in trajectories:
        g,po=terms(own);score=np.asarray(own)-prob
        direct+=po*float(np.dot(np.array([1.,gamma])*g,score))
        for peer in trajectories:
            gp,pp=terms(peer)
            baseline_effect+=po*pp*np.array([1.,gamma])*gp*score
            loo+=po*pp*float(np.dot(np.array([1.,gamma])*(g-gp),score))
    def expected(z):
        q=1/(1+np.exp(-z));total=0.
        for a,b in trajectories:
            r0=.2*a-.1;r1=1.3*b+.7*a*b
            total+=q**(a+b)*(1-q)**(2-a-b)*(r0+gamma*r1)
        return total
    numerical=(expected(theta+1e-5)-expected(theta-1e-5))/2e-5
    assert np.max(np.abs(baseline_effect))<1e-14 and abs(loo-direct)<1e-14 and abs(direct-numerical)<1e-9
    counts=[np.array([100,100]),np.array([100]),np.array([100,50]),np.array([100,100])]
    g=[np.array([3.,2.]),np.array([4.]),np.array([5.,1.]),np.array([2.,3.])]
    b=physical_time_baselines(g,counts,['x']*4);assert b[0][1]==(0+1+3)/3
    return dict(state='PASS',full_binary_trajectory_enumeration=True,independent_baseline_expected_score_max=float(np.max(np.abs(baseline_effect))),
        exact_policy_gradient=direct,numerical_gradient=numerical,baseline_gradient=loo,
        true_physical_time_and_absorbing_terminal_test=True,native_steps=0,optimizer_calls=0,
        limitation='Synthetic estimator identity only; no universal variance reduction or real navigation improvement claim.')


L.training_batch=training_batch
