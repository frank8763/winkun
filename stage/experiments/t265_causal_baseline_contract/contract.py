"""Fixed48 coefficient identities and exact finite-policy enumeration. No Actor."""
from pathlib import Path
import hashlib
import itertools
import json
import math
import shutil
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t265_causal_baseline_contract'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def coefficients(rewards, counts, baseline, gamma):
    ret = np.empty(len(rewards), dtype=np.float64)
    acc = 0.
    for k in range(len(rewards)-1, -1, -1):
        acc = float(rewards[k]) + gamma**int(counts[k]) * acc
        ret[k] = acc
    times = np.r_[0, np.cumsum(counts)[:-1]]
    discount = gamma**times
    # A_t baseline only uses other episodes' start returns and own past rewards.
    past = np.r_[0., np.cumsum(discount * rewards)[:-1]]
    causal_baseline = (baseline - past) / discount
    current_weight = discount * (ret - baseline)
    candidate_weight = discount * (ret - causal_baseline)
    expected = np.full(len(ret), ret[0] - baseline)
    return current_weight, candidate_weight, float(np.max(np.abs(candidate_weight-expected))), ret


def enumerated_contract():
    gamma = .91
    theta = np.array([.23, -.37])
    def sigmoid(x): return 1/(1+math.exp(-x))
    p0 = sigmoid(theta[0])
    paths = []
    for a0,a1 in itertools.product((0,1), repeat=2):
        p1 = sigmoid(theta[1] + .2*(2*a0-1) + .3*theta[0])
        prob = (p0 if a0 else 1-p0)*(p1 if a1 else 1-p1)
        score = np.array([[a0-p0, 0.], [.3*(a1-p1), a1-p1]])
        # Arbitrary causal potential, with terminal potential zero.
        phi0,phi1 = .21, .09+.17*a0
        event = 1. if a1 and a0 else (-1. if not a1 else -.5)
        rewards = np.array([phi0-gamma*phi1-.01, phi1-.01+event])
        value = rewards[0]+gamma*rewards[1]
        paths.append((prob,score,rewards,value))
    mean = sum(p*v for p,s,r,v in paths)
    exact = sum(p*v*s.sum(0) for p,s,r,v in paths)
    old = np.zeros(2)
    new = np.zeros(2)
    for p,s,r,v in paths:
        ow,nw,error,_ = coefficients(r,np.ones(2,dtype=int),mean,gamma)
        assert error < 1e-12
        old += p*(ow[:,None]*s).sum(0)
        new += p*(nw[:,None]*s).sum(0)
    assert np.max(np.abs(old-exact))<1e-12 and np.max(np.abs(new-exact))<1e-12
    # A different all-cap/no-event task has constant start return but varying potential.
    old_second = new_second = 0.
    for p,s,r,v in paths:
        phi1=.2+.03*float(s[0,0]+p0)
        rewards=np.array([.3-gamma*phi1-.01,phi1-.01-.5])
        constant=.3-.01+gamma*(-.01-.5)
        ow,nw,error,_=coefficients(rewards,np.ones(2,dtype=int),constant,gamma)
        old_second += p*np.square((ow[:,None]*s).sum(0)).sum()
        new_second += p*np.square((nw[:,None]*s).sum(0)).sum()
    assert old_second>0 and new_second<1e-25
    return dict(state='PASS',trajectory_count=4,conditional_logits_depend_on_history_and_both_parameters=True,
        expected_true_gradient=exact.tolist(),expected_current_estimator=old.tolist(),
        expected_candidate_estimator=new.tolist(),
        same_cap_constant_outcome_current_second_moment=float(old_second),
        same_cap_constant_outcome_candidate_second_moment=float(new_second),
        note='Exact finite toy, unnormalized likelihood-ratio gradients only; no general variance guarantee.')


def main():
    start=time.monotonic()
    assert not OUT.exists(), 'No duplicate or expanded contract run'
    plan=json.loads((HERE/'protocol.json').read_text())
    for name,h in plan['input_hashes'].items(): assert sha(Path(name).read_bytes())==h
    lines=Path(plan['training_log']).read_bytes().splitlines(keepends=True)[:48]
    assert len(lines)==48 and sha(b''.join(lines))==plan['prefix_sha256']
    rows=list(map(json.loads,lines))
    arrays=[]
    returns=[]
    gamma=math.exp(-.002/60)
    for r in rows:
        assert time.monotonic()-start<plan['wall_seconds']
        assert r['complete'] and sha(Path(r['path']).read_bytes())==r['sha256']
        with np.load(r['path']) as z: a={k:z[k].copy() for k in ['rewards','native_counts']}
        counts=a['native_counts'].astype(np.int64)
        assert int(counts.sum())==r['physics_steps']
        a['native_counts']=counts
        _,_,_,ret=coefficients(a['rewards'],counts,0.,gamma)
        arrays.append(a); returns.append(ret)
    groups={}
    for j,r in enumerate(rows): groups.setdefault(r['source_reset_id'],[]).append(j)
    reports=[]
    maxerror=0.
    for name,indices in groups.items():
        assert len(indices)==4 and len({rows[j]['policy_hash'] for j in indices})==1
        oldenergy=newenergy=0.
        for j in indices:
            baseline=float(np.mean([returns[k][0] for k in indices if k!=j]))
            a=arrays[j]
            old,new,error,_=coefficients(a['rewards'],a['native_counts'],baseline,gamma)
            maxerror=max(maxerror,error)
            oldenergy+=float(np.square(old).sum());newenergy+=float(np.square(new).sum())
        reports.append(dict(id=name,safe=sum(rows[j]['success'] and not rows[j]['collision'] for j in indices),
            contact=sum(rows[j]['collision'] for j in indices),timeout=sum(rows[j]['timeout'] for j in indices),
            complete_returns=[float(returns[j][0]) for j in indices],
            current_unscaled_coefficient_energy=oldenergy,candidate_unscaled_coefficient_energy=newenergy,
            candidate_to_current_energy_ratio=newenergy/max(oldenergy,1e-30)))
    assert maxerror<1e-11
    result=dict(state='COMPLETE',physics_steps=0,optimizer_calls=0,new_navigation_RL_updates=0,
        model_instantiations=0,episodes=48,groups=reports,max_coefficient_identity_error=maxerror,
        exact_enumeration=enumerated_contract(),elapsed_s=time.monotonic()-start,
        interpretation='Causal baseline coefficient contract, not policy improvement, calibrated risk, or trained new algorithm.',
        limitations=['Coefficient energy is not network-gradient variance.',
            'No reduction guarantee for arbitrary outcomes or policies.',
            'PPO clipping away from the behavior policy and random batch RMS normalization change estimator properties.',
            'Removing all-timeout finite-sample signal does not create left-side goal-reaching examples.',
            'Actor gradient comparison and an independently registered bounded navigation trial remain required.'])
    assert result['elapsed_s']<plan['wall_seconds']
    data=(json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode()
    assert len(data)<plan['max_output_bytes']
    assert sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())+len(data)<1024**3
    assert shutil.disk_usage(ROOT).free>=40*1024**3
    OUT.mkdir();(OUT/'summary.json').write_bytes(data)
    print(json.dumps({k:result[k] for k in ['state','physics_steps','optimizer_calls','episodes','max_coefficient_identity_error','elapsed_s']}))


if __name__=='__main__': main()
