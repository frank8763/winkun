"""Fixed 80+16 ended records only: availability/support, never new training."""
from pathlib import Path
import collections
import json
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from t271_runtime import *
DEST=ROOT/'stage/docs/long_distance_20261101/T271_NEXT_DATA_AUDIT.json'


def main():
    assert not DEST.exists()
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T271_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    paths=[ROOT/'stage/runs/t267_causal_credit_course/training.jsonl',ROOT/'stage/runs/t271_frozen_feature_exploration/exploration.jsonl']
    old=list(map(json.loads,paths[0].read_text().splitlines()))
    new=[r for r in map(json.loads,paths[1].read_text().splitlines()) if r['mode']=='feature_copula']
    assert len(old)==80 and len(new)==16
    signatures=set();policies=set();groups=collections.defaultdict(list);details=[];count=0
    for r in old+new:
        assert C.sha(r['path'])==r['sha256'] and r['complete']
        with np.load(r['path']) as z:
            for key,h in r['array_hashes'].items():assert S.digest(z[key])==h
            x=z['states'];nxt=z['next_states'];a=z['actions'];dt=z['native_counts'].astype(np.int64);flags=z['end_flags']
            assert x.shape==nxt.shape and x.shape[1]==120 and a.shape==(len(x),2)
            assert np.isfinite(x).all() and np.isfinite(nxt).all() and np.isfinite(a).all()
            assert np.array_equal(x[0,117:120],np.zeros(3,dtype=np.float32))
            assert np.array_equal(x[1:,117:119],a[:-1]) and np.array_equal(x[1:,119],dt[:-1].astype(np.float32)/100)
            assert np.array_equal(nxt[:,117:119],a) and np.array_equal(nxt[:,119],dt.astype(np.float32)/100)
            assert int(dt.sum())==r['physics_steps'] and np.all(flags[:-1]==0)
            assert np.array_equal(flags[-1],np.array([r['collision'],r['success'],r['timeout']],dtype=np.uint8))
            count+=len(x)
        key=(r['job']['case']['distance_m'],r['job']['case']['side']);groups[key].append(r)
        signature=(r['array_hashes']['states'],r['array_hashes']['actions'],r['integration_sha256'])
        assert signature not in signatures;signatures.add(signature);policies.add(r['policy_hash'])
        details.append(dict(id=r['id'],policy_hash=r['policy_hash'],path=r['path'],sha256=r['sha256'],
            historical_overlap_reported=bool(r.get('exact_historical_overlap')),distance_m=key[0],side=key[1]))
    assert count<=30000
    output=dict(state='READ_ONLY_INPUT_AVAILABILITY_AND_SUPPORT_AUDIT',physics_added=0,optimizer_added=0,
        records=96,decision_rows=count,distinct_behavior_policy_hashes=len(policies),unique_state_action_integration_hashes=len(signatures),
        source_rows_sha256={str(p):C.sha(p) for p in paths},source_sha256=C.sha(__file__),
        all_saved_array_sha_and_full120_finite_verified=True,previous_action_hold_context_exact=True,
        group_support=[dict(distance_m=k[0],side=k[1],records=len(v),safe=sum(r['success'] and not r['collision'] for r in v),
            contact=sum(r['collision'] for r in v),timeout=sum(r['timeout'] for r in v)) for k,v in sorted(groups.items())],
        records_detail=details,independent_training_seeds=0,
        historical_repeats_with_earlier_experiments_not_new_independent_data=sum(bool(r.get('exact_historical_overlap')) for r in old),
        future_goal_history_relabel_contract_verified=False,new_critic_trained=False,action_ranking_or_risk_calibration_demonstrated=False,
        limitation='The 6m records have no final-goal successes. Full states enable a representation pilot but do not identify calibrated 6m goal-reaching probabilities or make mixed-behavior replay an on-policy value. No automatic Actor update.')
    C.save(DEST,output);print(json.dumps({k:output[k] for k in ['state','records','decision_rows','distinct_behavior_policy_hashes','group_support']}),flush=True)


if __name__=='__main__':main()
