"""Posthoc descriptive common-goal labels on the three COMPLETED old paths.

No new targets are selected: only the21 eligible targets from the registered
source requests are reused across the three complete same-initial-state paths.
"""
from t277_labels import *
import math
OUT=ROOT/'stage/runs/t277_physical_hindsight'
DOC=ROOT/'stage/docs/long_distance_20261101'
DEST=DOC/'T277_CROSS_PATH_SUPPORT.json'


def main():
    assert not DEST.exists()
    audit=json.loads((DOC/'T277_PARTIAL_AUDIT.json').read_text())
    assert audit['state']=='PARTIAL_EVIDENCE_RECONCILED_NOT_FULL_RECOVERY'
    p=json.loads((OUT/'protocol.json').read_text());rows=list(map(json.loads,(OUT/'cases.jsonl').read_text().splitlines()))
    cache={};initials=[]
    for row,ref in zip(rows,p['references']):
        assert sha(row['path'])==row['sha256'] and row['collision'] and not row['success']
        with np.load(row['path']) as z:cache[row['id']]=(z['physical_xy'].copy(),z['physical_flags'].copy())
        with np.load(ref['path']) as z:initials.append((z['states'][0].copy(),z['poses'][0].copy()))
    assert len(rows)==3 and all(np.array_equal(a,initials[0][0]) and np.array_equal(b,initials[0][1]) for a,b in initials)
    targets=[r for r in audit['recovered_label_records'] if r['eligible']];assert len(targets)==21
    output=[];mixed=0
    for target in targets:
        goal=target['goal_xy'];labels=[]
        for row in rows:
            xy,flags=cache[row['id']]
            dd=np.fromiter((math.hypot(float(goal[0])-float(q[0]),float(goal[1])-float(q[1])) for q in xy),dtype=np.float64,count=len(xy))
            hit=np.flatnonzero((dd<=.2)&((flags&1)==0))
            if len(hit):
                first=int(hit[0])+1;assert not np.any(flags[:first]&1);outcome='GOAL_BEFORE_CONTACT'
            else:first=None;outcome='CONTACT_BEFORE_GOAL'
            labels.append(dict(path_id=row['id'],outcome=outcome,first_goal_step=first,observed_end_step=len(xy),
                original6m_outcome_unchanged='CONTACT'))
        kinds={r['outcome'] for r in labels};mixed+=len(kinds)>1
        output.append(dict(target_source_id=target['source_id'],target_chosen_decision=target['chosen_decision'],goal_xy=goal,
            mixed_path_outcomes=len(kinds)>1,labels=labels))
    save(DEST,dict(state='POSTHOC_COMMON_GOAL_LABEL_SUPPORT_DESCRIPTION',physics_added=0,optimizer_added=0,
        source_complete_paths=3,side='LEFT',unrecovered_RIGHT_not_included=True,
        original_initial120_and_pose_exact_across_three_paths=True,registered_eligible_targets_reused=21,
        goal_path_pairs=63,mixed_targets=mixed,
        goal_before_contact_pairs=sum(r['outcome']=='GOAL_BEFORE_CONTACT' for t in output for r in t['labels']),
        contact_before_goal_pairs=sum(r['outcome']=='CONTACT_BEFORE_GOAL' for t in output for r in t['labels']),targets=output,
        original6m_successes=0,new_physical_or_independent_samples=0,causal_first_action_ranking_passed=False,
        limitation='Posthoc description on only the completed biased-left subset; targets are training hindsight queries, not original6m successes. Whole stochastic continuations differ, so outcome variation is not a causal effect of the first action or a calibrated new-policyQ.'))
    print(json.dumps(dict(state='COMPLETE',common_goals=21,path_goal_pairs=63,mixed_targets=mixed)),flush=True)


if __name__=='__main__':main()
