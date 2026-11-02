"""Read-only rejected-value diagnostics. No fit, optimizer or native steps.

Preserve the preregistered raw screen. Centering and tie-aware calculations are
posthoc descriptions and never change whether the original pilot passed.
"""
from pathlib import Path
import sys, json, os
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t304_native_chunk_value'))
from t304_core import *
DEST=DOC/'T304_POSTHOC.json'


def main():
    assert not DEST.exists();p=read(HERE/'protocol.json');s=read(OUT/'summary.json')
    assert s['state']=='COMPLETE' and not s['screen']['passed'] and s['all_workers_stopped']
    w=read(OUT/'workers.json')
    for pid in [w['parent'],*w['workers'],read(OUT/'launch.json')['pid']]:
        try:os.kill(pid,0)
        except ProcessLookupError:continue
        raise AssertionError(('Live or reused PID',pid))
    raw=read(OUT/'cases.json');families=[];residual=0.
    for source in p['sources']:
        rows=[next(r for r in raw if r['family']==source['id'] and r['arm']==arm) for arm in p['arms']]
        desc=[]
        for r in rows:
            assert C.sha(r['path'])==r['sha256'];a=arrays(r['path']);b=source['branch_decision']
            steps=int(a['native_counts'][b:].astype(np.int64).sum())
            d0=float(np.linalg.norm(a['states'][b,51:53].astype(np.float64)*10))
            terminal_event=float(r['success'])-float(r['collision'])-.5*float(r['timeout'])
            analytic=.03*d0-.000002*(1-R.GAMMA**steps)/(1-R.GAMMA)+R.GAMMA**(steps-1)*terminal_event
            env=float(S.monte_carlo(a['environment_rewards'],a['native_counts'])[b])
            # The current observation stores float32 goal fields; use native
            # pose-derived goal distance for the exact potential telescoping.
            d0native=float(np.linalg.norm(np.asarray(r['goal_xy'])-a['poses'][b,:2]))
            analytic_native=.03*d0native-.000002*(1-R.GAMMA**steps)/(1-R.GAMMA)+R.GAMMA**(steps-1)*terminal_event
            error=abs(env-analytic_native);assert error<1e-11;residual=max(residual,error)
            cost=float(S.monte_carlo(a['aux_costs'],a['native_counts'])[b])
            desc.append(dict(arm=r['arm'],safe=r['success'] and not r['collision'],contact=r['collision'],timeout=r['timeout'],
                whole_episode_time_s=.002*r['physics_steps'],remaining_physics_from_branch=steps,
                remaining_m=r['remaining_m'],environment_G_at_branch=env,discounted_endpoint_cost=cost,
                eta01_G_at_branch=env-.1*cost,analytic_environment_G_native=analytic_native,
                float32_goal_field_formula_error=abs(env-analytic)))
        center=desc[1]
        for r in desc:
            r.update(environment_advantage=r['environment_G_at_branch']-center['environment_G_at_branch'],
                cost_advantage_contribution=-.1*(r['discounted_endpoint_cost']-center['discounted_endpoint_cost']),
                eta01_advantage=r['eta01_G_at_branch']-center['eta01_G_at_branch'])
        families.append(dict(id=source['id'],split=source['split'],rows=desc,
            outcome_type_changes_vs_center=sum((r['safe'],r['contact'],r['timeout'])!=(center['safe'],center['contact'],center['timeout']) for r in desc)))
    with np.load(OUT/'head_evidence.npz') as z:
        y=z['raw_y'].copy();test=~z['train'];posthoc={}
        for name in ['action','blind']:
            pred=z[name+'_prediction'];details=[];centered_error=[]
            for group in range(8):
                ix=np.arange(group*3,group*3+3)
                if not test[ix].all():continue
                truth=y[ix];guess=pred[ix];centered=guess-guess[1];err=(centered-truth)**2
                maximizers=guess==guess.max();valid=truth.max()-truth<=p['value_margin']
                tiecredit=float(np.sum(maximizers&valid)/maximizers.sum())
                details.append(dict(family=p['sources'][group]['id'],predictions_center_removed=centered.tolist(),
                    true_advantages=truth.tolist(),posthoc_centered_MSE=float(err.mean()),
                    maximizer_count=int(maximizers.sum()),tie_aware_top_choice_credit=tiecredit))
                centered_error.extend(err.tolist())
            posthoc[name]=dict(centered_test_MSE=float(np.mean(centered_error)),details=details)
        losses=z['losses'].copy()
    m=read(OUT/'metrics.json')
    result=dict(scope='Stopped read-only posthoc; original protocol/metrics/failed screen untouched, no new fit/Actor/optimizer/native',
        native_steps=0,optimizer_calls=0,Actor_updates=0,original_screen=s['screen'],
        raw_action_test_MSE_over_blind=m['action']['test_MSE']/m['blind']['test_MSE'],
        initial_normalized_training_losses=losses[0].tolist(),final_normalized_training_losses=losses[-1].tolist(),
        posthoc_same_state_centering=posthoc,
        centering_caveat='Advantage regression has nonzero prediction for a=center on new states. Subtracting predicted center is a posthoc check only: it removes additive bias but not wrong pairwise ranking. Centered aggregate action MSE still exceeds centered blind. Original failed gate is not redefined.',
        tie_caveat='Original top_choice_correct uses deterministic argmax (first in array). Blind gives exact ties for all3 actions; its apparent LEFT top success is only first-index tie breaking. Tie-aware credit is1/3 on each family; do not call blind an actionable winning policy.',
        families=families,environment_return_telescope_max_error=residual,
        reward_caveat='Within a fixed branch state and identical terminal event, original potential-shaped environment return depends only on remaining time to that event; left perturbations all end in CONTACT, so a higher return can partly mean a delayed collision. eta.1 endpoint cost is retained, not new shaping. This is an objective property, not a new reward implementation bug or proof allRL cannot learn.',
        source_sha256=C.sha(__file__),summary_sha256=C.sha(OUT/'summary.json'))
    C.save(DEST,result);print(json.dumps(dict(state='POSTHOC_COMPLETE',raw_MSE_ratio=result['raw_action_test_MSE_over_blind'],
        centered_action_MSE=posthoc['action']['centered_test_MSE'],centered_blind_MSE=posthoc['blind']['centered_test_MSE'],
        environment_telescope_max_error=residual,outcome_type_changes=sum(f['outcome_type_changes_vs_center'] for f in families))))


if __name__=='__main__':main()
