"""Independent exact math.hypot radius predicate used by the native goal state."""
from t277_labels import *
import math
OUT=ROOT/'stage/runs/t277_physical_hindsight'
DEST=ROOT/'stage/docs/long_distance_20261101/T277_NATIVE_RADIUS_CHECK.json'


def main():
    assert not DEST.exists()
    assert json.loads((ROOT/'stage/docs/long_distance_20261101/T277_AUDIT.json').read_text())['state']=='POSTRUN_EVIDENCE_RECONCILED'
    rows={r['id']:r for r in map(json.loads,(OUT/'cases.jsonl').read_text().splitlines())}
    labels=json.loads((OUT/'hindsight_first_hits.json').read_text());results=[]
    cache={}
    for r in labels:
        if not r['eligible']:continue
        name=r['source_id']
        if name not in cache:
            with np.load(rows[name]['path']) as z:cache[name]=(z['physical_xy'].copy(),z['physical_flags'].copy())
        xy,flags=cache[name];goal=r['goal_xy']
        distance=np.fromiter((math.hypot(float(goal[0])-float(p[0]),float(goal[1])-float(p[1])) for p in xy),dtype=np.float64,count=len(xy))
        candidates=np.flatnonzero((distance<=.2)&((flags&1)==0));assert len(candidates)
        first=int(candidates[0])+1;assert first==r['first_goal_step'] and not np.any(flags[:first]&1)
        # First native point must be strictly after original initial; the
        # independent main audit checks the original initial radius separately.
        before=float(distance[first-2]) if first>1 else r['distance_before_m']
        assert before>.2 and distance[first-1]<=.2
        results.append(dict(source_id=name,chosen_decision=r['chosen_decision'],first_goal_step=first,
            before_margin_m=before-.2,hit_margin_m=.2-float(distance[first-1])))
    save(DEST,dict(state='NATIVE_MATH_HYPOT_FIRSTHIT_PREDICATE_VERIFIED',physics_added=0,optimizer_added=0,
        all_eligible_labels_match_native_math_hypot=True,records=results,
        physical_XY_payloads_referenced_not_simulated_again=True,firsthit_count=len(results)))
    print(json.dumps(dict(state='PASS',firsthit_count=len(results))),flush=True)


if __name__=='__main__':main()
