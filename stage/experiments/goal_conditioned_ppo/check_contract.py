"""Zero-native, zero-fit identities on all completed T310 diagnostic rows.

The sinusoidal V is deliberately synthetic: this checks algebra and terminal
semantics, not critic accuracy, variance reduction, or policy improvement.
"""
from pathlib import Path
import sys,json
import numpy as np
from advantage import advantage
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t309_sequence_goal_value'))
from sequence_value import B,R

def main():
    rows=[]
    for folder in ['t310_initial_history_control','t310_initial_history_recovery']:
        rows+=json.loads((ROOT/'stage/runs'/folder/'cases.json').read_text())
    assert len(rows)==20
    maximum=0.;censored_rejections=0
    for row in rows:
        a=B.arrays(row['path']);n=len(a['rewards']);v=np.sin(np.arange(n+1)*.37)
        result=advantage(a['rewards'],a['native_counts'],v,a['end_flags'],gamma=R.GAMMA,lam=1.)
        mc=np.empty(n);carry=0.
        for k in range(n-1,-1,-1):carry=float(a['rewards'][k])+R.GAMMA**int(a['native_counts'][k])*carry;mc[k]=carry
        err=float(np.max(np.abs(result['advantage']-(mc-v[:-1]))));maximum=max(maximum,err);assert err<1e-12
        zero=advantage(a['rewards'],a['native_counts'],v,a['end_flags'],gamma=R.GAMMA,lam=0.)
        assert np.array_equal(zero['advantage'],zero['delta'])
        other=v.copy();other[-1]+=1000
        again=advantage(a['rewards'],a['native_counts'],other,a['end_flags'],gamma=R.GAMMA)
        original=advantage(a['rewards'],a['native_counts'],v,a['end_flags'],gamma=R.GAMMA)
        assert np.array_equal(again['advantage'],original['advantage'])
        try:advantage(a['rewards'],a['native_counts'],v,a['end_flags'],gamma=R.GAMMA,complete=False)
        except ValueError:censored_rejections+=1
        else:raise AssertionError('Administrative censoring treated as terminal')
    print(json.dumps(dict(state='ENGINEERING_IDENTITIES_PASS',trajectories=20,lambda1_MC_identity_max_error=maximum,
        lambda0_equals_TD=True,true_cap_terminal_bootstrap_zero=True,administrative_censor_rejections=censored_rejections,
        native_steps=0,optimizer_calls=0,Actor_updates=0,trained_critic=False,scientific_control_validated=False)))
if __name__=='__main__':main()
