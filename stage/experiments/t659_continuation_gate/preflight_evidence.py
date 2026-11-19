"""Meaningful synthetic counterexamples for the offline evidence gate only."""
from pathlib import Path
import copy
import json
import numpy as np
import t659_evidence as E

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT/'stage/docs/day_20261119/t659'


def main():
    query = dict(query_key='fixture',id='synthetic',prefix_first=1,source_category='synthetic_not_Stage',
                 case=dict(step_cap=300),remaining_native_steps=200,candidate_action=[.5,.25])
    base = dict(states=np.zeros((3,120)),poses=np.zeros((3,3)),history_poses=np.zeros((3,1,3)),
                history_goals=np.zeros((3,1,2)),history_times=np.zeros((3,1)),actions=np.zeros((3,2),np.float32),
                native_counts=np.asarray([100,100,100]),next_states=np.zeros((3,120)),next_poses=np.zeros((3,3)),
                end_flags=np.asarray([[0,0,0],[0,0,0],[0,1,0]],np.uint8),
                initial_native_integration=np.zeros(203),task_goal_xy=np.ones(2),scene_sha256=np.zeros(32,np.uint8))
    arrays = {kind:copy.deepcopy(base) for kind in E.KINDS}
    for kind in ['candidate_once','candidate_full']:
        arrays[kind]['actions'][1] = [.5,.25]
    # Equal first native hold, but only the future controller fails.
    arrays['candidate_full']['end_flags'][2] = [1,0,0]
    arrays['candidate_full']['actions'][2] = [.6,.35]
    arrays['candidate_full']['poses'][2,0] = .03
    rows = [dict(complete=True,administrative_censor=False,success=kind!='candidate_full',collision=kind=='candidate_full',
                 timeout=False,remaining_m=.2 if kind!='candidate_full' else .8,path=kind,
                 request=dict(job=dict(query,continuation_kind=kind))) for kind in E.KINDS]
    load = lambda row:arrays[row['path']]
    p = dict(queries=[query])
    result = E.build_panel(p,rows,load)
    answer = result['queries'][0]
    assert answer['proxy_false_acceptance'] and answer['full_candidate_regression']
    assert answer['local_old_continuation_difference']==0 and answer['complete_policy_difference']==-1
    assert answer['suffix_replacement_effect']==-1 and answer['fork']['same_candidate_command_and_first_hold']
    assert answer['divergence']['first_different_post_fork_command']['decision']==2
    assert result['evidence_state']=='LOCAL_PROXY_MISMATCH_OBSERVED' and not result['automatic_learning']
    censored = copy.deepcopy(rows)
    censored[-1].update(complete=False,administrative_censor=True)
    value = E.build_panel(p,censored,load)
    assert value['evidence_state']=='INCOMPLETE_PANEL'
    assert value['queries'][0]['complete_policy_difference'] is None
    assert value['queries'][0]['outcomes']['candidate_full']['terminal']=='CENSORED'
    rejected=[]
    def rejects(label, fn):
        try:
            fn()
        except AssertionError:
            rejected.append(label)
        else:
            raise AssertionError('Invalid evidence accepted: '+label)
    rejects('duplicate_query_arm',lambda:E.build_panel(p,rows+[rows[0]],load))
    bad = copy.deepcopy(arrays)
    bad['candidate_full']['next_states'][1,0]=1.
    rejects('different_first_hold_physics',lambda:E.build_panel(p,rows,lambda r:bad[r['path']]))
    bad = copy.deepcopy(arrays)
    bad['candidate_once']['history_times'][1,0]=1.
    rejects('different_causal_fork',lambda:E.build_panel(p,rows,lambda r:bad[r['path']]))
    bad = copy.deepcopy(arrays)
    bad['candidate_full']['end_flags'][0]=[1,0,0]
    rejects('terminated_prefix',lambda:E.build_panel(p,rows,lambda r:bad[r['path']]))
    rejects('remaining_horizon_reset',lambda:E.build_panel(dict(queries=[dict(query,remaining_native_steps=300)]),rows,load))
    value=dict(state='PASS_SYNTHETIC_EVIDENCE_GATE_ONLY',new_native=0,new_optimizer=0,
               matched_first_hold_suffix_effect_classified=True,censor_kept_unknown=True,
               invalid_evidence_rejected=rejected,automatic_learning=False,
               navigation_result=False,actual_branch_responses_not_measured=True)
    DOC.mkdir(parents=True,exist_ok=True)
    (DOC/'EVIDENCE_PREFLIGHT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':
    main()
