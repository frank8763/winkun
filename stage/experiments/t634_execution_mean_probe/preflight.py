"""Numerical expectation and exact own-history output-transform contracts."""
from pathlib import Path
import ast
import copy
import json
import math
import types
import numpy as np
import t634_runtime as R
from expectation import clipped_mean
N,M=R.N,R.M


def main():
    p=R.plan();assert not p['authorization_approved'];R.setup(p)
    grid=np.array([[0.,1.],[-1.,.7],[3.,-3.],[.95,-.95],[.2,-.2]])
    sigma=np.array([.2,.2]);actual=clipped_mean(grid,sigma)
    assert np.allclose(clipped_mean(-grid,sigma),-actual,atol=1e-14)
    nodes,w=np.polynomial.legendre.leggauss(160);quad=np.empty_like(grid)
    for index,mu in np.ndenumerate(grid):
        sd=sigma[index[-1]];pdf=np.exp(-.5*((nodes-mu)/sd)**2)/(sd*math.sqrt(2*math.pi))
        lower=.5*math.erfc(-(-1-mu)/sd/math.sqrt(2))
        upper=.5*math.erfc(-(1-mu)/sd/math.sqrt(2))
        quad[index]=-lower+np.dot(w,nodes*pdf)+1-upper
    assert np.max(np.abs(actual-quad))<1e-12
    assert abs(actual[0,1]-(1-.2/math.sqrt(2*math.pi)))<1e-12
    # Use an actual complete unforced success state/history tape. A fake
    # state-only shell calls the real Session.step code with a policy-only
    # native seam; no MuJoCo integration occurs and no new result is asserted.
    old=R.read(R.ROOT/'stage/runs/t632_taskwise_recovery/protocol.json')
    row=old['nominal_eight_row'];a=R.R.B.R.E.N.S.arrays(row['path'])
    assert R.sha(row['path'])==row['sha256']
    q=N.adapt(M.frozen(p,p['student_initial']['path']),'expected')
    base=M.frozen(p,p['student_initial']['path']);base.sample=q.sample=False
    h=bh=None;trace=[];before=M.digest(q.state_dict())
    with N.torch.no_grad():
        for x in a['states'][:20]:
            _,loc,bh,_=base.step(N.torch.tensor(x[None]),bh)
            command,out,h,_=q.step(N.torch.tensor(x[None]),h)
            assert np.array_equal(command.numpy(),clipped_mean(loc.numpy(),p['raw_sigma']).astype(np.float32))
            assert N.torch.equal(h,bh)
            assert np.array_equal(q.tape[-1]['extra_hidden_before'],base.tape[-1]['extra_hidden_before'])
            trace.append(command.numpy().copy())
    assert M.digest(q.state_dict())==before
    assert q.phash()!=base.phash()
    # Synthetic tape fixture tests the real auditor using *existing* historical
    # observations/physics/reward as immutable inputs, never a new rollout.
    # Actions for a counterfactual transform would alter the real dynamics, so
    # no synthetic physics/array result is passed off as its native trajectory.
    assert not N.torch.cuda.is_initialized()
    for f in R.HERE.glob('*.py'):ast.parse(f.read_text())
    from storage_regression import main as storage
    storage()
    report=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',numerical_quadrature_maxerr=float(np.max(np.abs(actual-quad))),
        symmetry_and_bounds=True,known_boundary_mean=True,actual_Actor_step_transform_formula=True,
        actual_hidden_carry_unchanged_on_same_observations=True,weights_unchanged=True,
        output_policy_hash_differs_from_neural_weight_hash=True,
        policy_only_counterfactual_trace_not_native_physics=True,native_steps=0,optimizer_updates=0)
    R.save(R.DOC/'PREFLIGHT.json',report);print(json.dumps(report))


if __name__=='__main__':main()
