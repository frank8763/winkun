"""Independent event reconstruction and every-step frozen continuation check."""
from t273_native import *
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t273_value_direction_probe'
DEST=ROOT/'stage/docs/long_distance_20261101/T273_EFFECTS.json'


def main():
    assert not DEST.exists()
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T273_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    p=json.loads((OUT/'protocol.json').read_text());q=policy(p).eval().requires_grad_(False)
    rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()))[::2];items=[];n=0
    for r in rows:
        with np.load(r['path']) as z:
            counts=z['native_counts'].astype(np.int64);flags=z['end_flags'];native_n=int(counts.sum())
            expected=float(flags[-1,1])-float(flags[-1,0]) if native_n<=8000 else 0.
            assert expected==r['actual_score16']
            for x,a in zip(z['states'][1:],z['actions'][1:]):
                with torch.no_grad():actual=q.step(torch.tensor(x[None]))[0][0].numpy()
                assert np.array_equal(actual,a);n+=1
        items.append(dict(action=r['proposal'],first_action=p['proposals']['actions'][r['proposal']],
            predicted_U16=r['predicted_score16'],actual_U16=expected,
            time_until_terminal_s=native_n*.002,absolute_terminal_s=r['absolute_end_step']*.002,
            GOAL=r['success'],CONTACT=r['collision'],TIMEOUT=r['timeout'],remaining_m=r['remaining_m']))
    labels=[r['actual_U16'] for r in items];pairs=sum(labels[i]!=labels[j] for i in range(3) for j in range(i+1,3))
    C.save(DEST,dict(state='RAW_EVENT_AND_FROZEN_CONTINUATION_RECONSTRUCTED',physics_added=0,optimizer_added=0,
        every_noninitial_action_matches_frozen_Actor_bitwise=n,unique_action_results=items,
        empirically_different_score16_pairs=pairs,not_new_training_or_stable_navigation=True,
        interpretation='Terminal latency/remaining distance are descriptive. A tied sixteen-second outcome does not validate model action ranking, and one known deterministic context is not expected-value calibration.'))
    print(json.dumps(dict(state='COMPLETE',outcomes=items,informative_pairs=pairs)),flush=True)


if __name__=='__main__':main()
