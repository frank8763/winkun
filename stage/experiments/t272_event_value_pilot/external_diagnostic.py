"""Read-only six known T269 suffixes at two equal-state action contexts.

These previously consumed diagnostics were not fitting targets for T272. They
are not a new independent navigation holdout or evidence of causal rankings.
"""
from event_model import *
import gzip
import io
HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t272_event_value_pilot'
DEST=ROOT/'stage/docs/long_distance_20261101/T272_EXTERNAL_DIAGNOSTIC.json'


def main():
    assert not DEST.exists()
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T272_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    payload=torch.load(io.BytesIO(gzip.decompress((OUT/'event_checkpoint.pt.gz').read_bytes())),map_location='cpu',weights_only=False)
    q=EventModel();q.load_state_dict(payload['model']);q.eval()
    path=ROOT/'stage/runs/t269_same_state_controls/cases.jsonl'
    rows=list(map(json.loads,path.read_text().splitlines()))[2:];assert len(rows)==6
    result=[];groups={}
    p=json.loads((OUT/'protocol.json').read_text())
    for r in rows:
        assert sha(r['path'])==r['sha256']
        with np.load(r['path']) as z:
            context=z['states'][0].copy();action=z['actions'][0].copy()
            y,m=native_targets(z['native_counts'],z['end_flags'],28000-r['start_step'],r['complete'],p['horizons_physics'])
        x=np.concatenate([context,action]);xx=np.clip((x-payload['normalization_mean'])/payload['normalization_scale'],-10,10)
        with torch.no_grad():pred=q(torch.tensor(xx[None])).numpy()[0]
        item=dict(id=r['id'],start_step=r['start_step'],action=action.tolist(),true_first_event_labels=y[0].tolist(),
            valid_horizons=m[0].tolist(),predicted_probabilities=pred.tolist(),
            context120_sha256=hashlib.sha256(context.tobytes()).hexdigest(),source_payload_sha256=r['sha256'])
        result.append(item);groups.setdefault(r['start_step'],[]).append(item)
    contrasts=[]
    for start,gg in groups.items():
        assert len(gg)==3 and len({r['context120_sha256'] for r in gg})==1
        labels=np.array([r['true_first_event_labels'] for r in gg]);valid=np.array([r['valid_horizons'] for r in gg],dtype=bool)
        assert np.array_equal(valid, np.broadcast_to(valid[0],valid.shape))
        informative=sum(any(not np.array_equal(labels[a,h],labels[b,h]) for h in np.flatnonzero(valid[0])) for a in range(3) for b in range(a+1,3))
        contrasts.append(dict(start_step=start,pairs=3,empirically_distinguishable_event_ranking_pairs=int(informative),
            goal_probability_spread_per_horizon=np.ptp(np.array([r['predicted_probabilities'] for r in gg])[:,:,0],axis=0).tolist(),
            contact_probability_spread_per_horizon=np.ptp(np.array([r['predicted_probabilities'] for r in gg])[:,:,1],axis=0).tolist()))
    assert sum(x['empirically_distinguishable_event_ranking_pairs'] for x in contrasts)==0
    save(DEST,dict(state='KNOWN_COUNTERFACTUAL_LABEL_SUPPORT_AUDIT',physics_added=0,optimizer_added=0,
        source_rows_sha256=sha(path),event_checkpoint_sha256=sha(OUT/'event_checkpoint.pt.gz'),records=result,contexts=contrasts,
        independent_navigation_cases=1,same_native_state_contexts=2,counterfactual_action_ranking_passed=False,
        limitation='All three branches at each context have identical event classes at every valid registered horizon. These six known suffixes cannot validate a useful action ordering; predicted differences are not causal evidence. No Actor coupling.'))
    print(json.dumps(dict(state='COMPLETE',informative_action_pairs=0)),flush=True)


if __name__=='__main__':main()
