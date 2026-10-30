"""Post-hoc analytic action-support audit, zero physics and optimization."""
import os
os.sched_setaffinity(0,{11})
from pathlib import Path
import sys,json,math
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'stage/runs/t202_cause_audit'
sys.path.insert(0,str(ROOT/'stage/experiments/t200_residual_placement'))
import common as C
import numpy as np
import torch

def cdf(x):return .5*math.erfc(-float(x)/math.sqrt(2))
def main():
    assert not (OUT/'exploration_support.json').exists()
    torch.set_num_threads(1)
    plan=json.loads((ROOT/'stage/experiments/t200_residual_placement/protocol.json').read_text())
    path=ROOT/'stage/runs/t199_sac_mean_gradient/initial_residual_policy.pt';ck=torch.load(path,map_location='cpu',weights_only=False);p=C.Policy('latent').eval().requires_grad_(False);p.load_state_dict(ck['policy'])
    # This is the actual training reset sigma, not an assumed saved sigma.
    sigma=np.asarray(plan['initial_std'],dtype=float)
    with np.load(ROOT/'stage/runs/t196_left_exploration/left_baseline_trace.npz') as z:common=z['states'].copy()
    with torch.no_grad():mu=p.latent_mean(torch.tensor(common)).numpy().astype(float)
    saturated=mu[:,0]>=np.arctanh(np.float32(.999999))-.0001
    thresholds=[.9,.5,0.]
    probabilities={str(t):[cdf((np.arctanh(t)-m)/sigma[0]) for m in mu[saturated,0]] for t in thresholds}
    parent={'common_states':len(common),'near_positive_clip_states':int(saturated.sum()),'fraction_near_positive_clip':float(saturated.mean()),'training_sigma_latent':sigma.tolist(),'latent_throttle_mean_range_on_clip':[float(mu[saturated,0].min()),float(mu[saturated,0].max())],
            'probability_native_throttle_below':{t:{'min':min(v),'max':max(v)} for t,v in probabilities.items()},
            'central95_native_throttle_range_at_minimum_clip_mu':np.tanh(mu[saturated,0].min()+sigma[0]*np.asarray([-1.95996398454,1.95996398454])).tolist()}
    rows=[json.loads(s) for s in (ROOT/'stage/runs/t200_residual_placement/episodes.jsonl').read_text().splitlines()];observed={};hashes={str(path):C.sha(path)}
    for group in ['original_LEFT','near_LEFT']:
        subset=[r for r in rows if r['arm']=='native' and '_b0_' in r['id'] and bool(r['snapshot_side'])==(group=='near_LEFT')]
        states=[];actions=[]
        for row in subset:
            assert C.sha(row['path'])==row['sha256'];hashes[row['path']]=row['sha256']
            with np.load(row['path']) as z:states.append(z['states']);actions.append(z['actions'])
        states=np.concatenate(states);actions=np.concatenate(actions)
        with torch.no_grad():m=p.latent_mean(torch.tensor(states)).numpy();det=p(torch.tensor(states)).numpy()
        mask=m[:,0]>=np.arctanh(np.float32(.999999))-.0001
        observed[group]={'episodes':len(subset),'decisions':len(states),'positive_clip_decisions':int(mask.sum()),'positive_clip_fraction':float(mask.mean()),'observed_throttle_below_0_9_on_positive_clip':int(np.sum(actions[mask,0]<.9)),
                         'observed_throttle_range_on_positive_clip':None if not mask.any() else [float(actions[mask,0].min()),float(actions[mask,0].max())],
                         'deterministic_throttle_min_max':[float(det[:,0].min()),float(det[:,0].max())]}
    # Complete environment return ordering, using the exact project's contract.
    gamma=math.exp(-.002/60)
    def ret(seconds,event,d=5):
        n=round(seconds/.002)
        return -.000002*(1-gamma**n)/(1-gamma)+gamma**(n-1)*event+.03*d
    failure=ret(30.064,-1);timeout=ret(46.668,-.5);success=ret(42.794,1)
    result={'state':'COMPLETE','analysis_type':'POST_HOC_ANALYTIC_AND_SAVED_BEHAVIOR_ONLY','physics_steps':0,'optimizer_updates':0,'input_hashes':hashes,'parent_on_common_LEFT':parent,'actual_first_native_behavior_batch':observed,
            'environment_objective_example':{'initial_distance_m':5,'collision_at_30_064s_return':failure,'timeout_at_46_668s_return':timeout,'hypothetical_GOAL_at_RIGHT_reference_time_42_794s_return':success,'hypothetical_success_probability_needed_to_beat_safe_timeout_if_otherwise_collision':(timeout-failure)/(success-failure),'scope':'Analytic event/time examples, not an observed LEFT success or estimate of reachable policy value. Terminal-invariant potential leaves same-state/same-time timeout trajectories equal regardless of final distance; T197 distance penalty already tested without LEFT success.'},
            'scope':'Inverse-squash warm start can collapse physical throttle exploration despite latent sigma.5. Restoring a latent mean gradient leaves initial sampled action distribution unchanged; it does not restore broad physical action coverage. Final models may leave this regime, so this is not sufficient proof of sole failure cause. Extra T196 parameter noise affected steering only.'}
    C.save(OUT/'exploration_support.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='input_hashes'},indent=2))

if __name__=='__main__':main()
