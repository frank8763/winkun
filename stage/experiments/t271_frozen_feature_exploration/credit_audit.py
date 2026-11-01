"""Read-only exact return/trajectory-coefficient audit on the fixed 32 probes."""
from pathlib import Path
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from t271_runtime import *
OUT=ROOT/'stage/runs/t271_frozen_feature_exploration'
DEST=ROOT/'stage/docs/long_distance_20261101/T271_CREDIT_AUDIT.json'


def main():
    assert not DEST.exists()
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T271_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    rows=list(map(json.loads,(OUT/'exploration.jsonl').read_text().splitlines()))
    geo={r['id']:r for r in map(json.loads,(OUT/'geometry.jsonl').read_text().splitlines())}
    records=[]
    for r in rows:
        assert C.sha(r['path'])==r['sha256']
        with np.load(r['path']) as a:
            g=float(S.monte_carlo(a['rewards'],a['native_counts'])[0])
            d0=float(np.linalg.norm(a['poses'][0,:2]-np.asarray(geo[r['source_reset_id']]['goal_xy'])))
        n=r['physics_steps'];event=float(r['success'])-float(r['collision'])-.5*float(r['timeout'])
        exact=.03*d0-.000002*(1-R.GAMMA**n)/(1-R.GAMMA)+R.GAMMA**(n-1)*event
        assert abs(g-exact)<1e-12
        records.append(dict(id=r['id'],source_reset_id=r['source_reset_id'],mode=r['mode'],seed=r['seed'],
            G0=g,closed_form=exact,closed_form_error=abs(g-exact),remaining_m=r['remaining_m'],
            collision=r['collision'],timeout=r['timeout'],physics_steps=n))
    groups=[]
    for name in geo:
        for arm in ['conditional_AR','feature_copula']:
            a=[r for r in records if r['source_reset_id']==name and r['mode']==arm]
            assert len(a)==4
            g=np.array([r['G0'] for r in a]);baseline=(g.sum()-g)/3;coeff=g-baseline
            timeouts=[r for r in a if r['timeout']]
            groups.append(dict(source_reset_id=name,mode=arm,records=4,timeout_records=len(timeouts),
                G0_range=float(np.ptp(g)),leave_one_out_starting_return_coefficient_max=float(np.abs(coeff).max()),
                timeout_G0_range=float(np.ptp([r['G0'] for r in timeouts])) if timeouts else None,
                timeout_remaining_m_range=[min(r['remaining_m'] for r in timeouts),max(r['remaining_m'] for r in timeouts)] if timeouts else None))
    C.save(DEST,dict(state='EXACT_RETURN_DESCRIPTION_NOT_NEW_TRAINING',physics_added=0,optimizer_added=0,
        original_reward_unchanged=True,maximum_closed_form_error=max(r['closed_form_error'] for r in records),
        records=records,groups=groups,
        interpretation='At a fixed start all full-cap timeouts have identical total return. Under the T267 causal baseline their weighted per-time coefficients equal G0-minus-other-three mean. This leaves no distance-ranking signal among those timeouts. Mixed collision/timeout groups can still have a signal. This is a known reward-objective property, not proof of the unique failure cause or an implementation bug.',
        do_not_repeat_T197_terminal_distance_penalty_grid=True))
    print(json.dumps(dict(state='COMPLETE',groups=groups)),flush=True)


if __name__=='__main__':main()
