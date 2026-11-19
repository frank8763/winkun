"""Per-query evidence, with explicit censoring and matched first-hold checks."""
import numpy as np

KINDS = ('reference', 'candidate_once', 'candidate_full')


def complete(row):
    return bool(row['complete'] and not row['administrative_censor'])


def safe(row):
    return bool(complete(row) and row['success'] and not row['collision']
                and not ('fullfield_start_pose' in row['request']['job'] and row.get('field_boundary', False)))


def terminal(row):
    if not complete(row):
        return 'CENSORED'
    if row['collision']:
        return 'CONTACT'
    if row.get('field_boundary', False) and 'fullfield_start_pose' in row['request']['job']:
        return 'BOUNDARY_AT_RECORDED_ENDPOINTS'
    if safe(row):
        return 'GOAL'
    if row['timeout']:
        return 'TIMEOUT'
    return 'OTHER_COMPLETE_FAILURE'


def matched_fork(query, arrays):
    k = query['prefix_first']
    reference, once, whole = [arrays[kind] for kind in KINDS]
    for a in arrays.values():
        assert len(a['states']) > k and not a['end_flags'][:k].any(), 'Prefix already terminated'
        assert int(a['native_counts'][:k].sum()) == query['case']['step_cap']-query['remaining_native_steps']
        for key in ['initial_native_integration', 'task_goal_xy', 'scene_sha256']:
            assert np.array_equal(a[key], reference[key]), ('Unmatched task', key)
        for key in ['states', 'poses', 'history_poses', 'history_goals', 'history_times']:
            assert np.array_equal(a[key][:k+1], reference[key][:k+1]), ('Unmatched causal fork', key)
        assert np.array_equal(a['actions'][:k], reference['actions'][:k])
        assert np.array_equal(a['native_counts'][:k], reference['native_counts'][:k])
    # The two arms must perform exactly the same native first hold before
    # attributing any later difference to replacing their future controllers.
    for key in ['actions', 'native_counts', 'next_states', 'next_poses', 'end_flags']:
        assert np.array_equal(once[key][k], whole[key][k]), ('First native hold changed', key)
    expected = np.asarray(query['candidate_action'], np.float32)
    assert np.array_equal(once['actions'][k], expected)
    return dict(prefix_native_steps=int(reference['native_counts'][:k].sum()),
                remaining_native_steps=query['remaining_native_steps'],
                first_hold_native_steps=int(once['native_counts'][k]),
                same_candidate_command_and_first_hold=True,
                reference_command=reference['actions'][k].tolist(), candidate_command=expected.tolist())


def divergence(once, whole, first):
    n = min(len(once['states']), len(whole['states']))
    before = [np.r_[0, np.cumsum(a['native_counts'])[:-1]][:n] for a in [once, whole]]
    aligned = before[0] == before[1]
    command_diff = np.any(once['actions'][:n] != whole['actions'][:n], axis=1) & aligned
    xy = np.linalg.norm(once['poses'][:n, :2]-whole['poses'][:n, :2], axis=1)
    dy = once['poses'][:n, 2]-whole['poses'][:n, 2]
    yaw = np.abs(np.arctan2(np.sin(dy), np.cos(dy)))*180/np.pi
    pose_diff = ((xy >= .02) | (yaw >= 1.)) & aligned
    def event(mask):
        choices = np.flatnonzero(mask & (np.arange(n) > first))
        if not len(choices):
            return None
        k = int(choices[0])
        return dict(decision=k, native_steps=int(before[0][k]), seconds=float(before[0][k]*.002),
                    xy_m=float(xy[k]), yaw_degrees=float(yaw[k]))
    return dict(first_different_post_fork_command=event(command_diff),
                first_post_fork_2cm_or_1degree=event(pose_diff),
                descriptive_not_a_global_stability_proof=True)


def build_panel(p, rows, load):
    measured = [r for r in rows if not r.get('exact')]
    index = {}
    for row in measured:
        key = (row['request']['job']['query_key'], row['request']['job']['continuation_kind'])
        assert key not in index, ('Duplicate query/continuation', key)
        index[key] = row
    assert len(index) == 3*len(p['queries'])
    answers = []
    for query in p['queries']:
        rr = {kind:index[(query['query_key'], kind)] for kind in KINDS}
        arrays = {kind:load(row) for kind, row in rr.items()}
        fork = matched_fork(query, arrays)
        known = all(complete(r) for r in rr.values())
        b, one, whole = [int(safe(rr[k])) for k in KINDS]
        answers.append(dict(query_key=query['query_key'], id=query['id'], prefix_first=query['prefix_first'],
            original_start=query['prefix_first'] == 0, source_category=query['source_category'], complete=known,
            reference=b if known else None, first_action_then_reference=one if known else None,
            full_candidate=whole if known else None,
            local_old_continuation_difference=one-b if known else None,
            complete_policy_difference=whole-b if known else None,
            suffix_replacement_effect=whole-one if known else None,
            first_action_proxy_mismatch=bool(known and one != whole),
            proxy_false_acceptance=bool(known and one and not whole),
            full_candidate_regression=bool(known and b and not whole),
            full_candidate_gain=bool(known and not b and whole),
            fork=fork, divergence=divergence(arrays['candidate_once'], arrays['candidate_full'], query['prefix_first']),
            outcomes={kind:dict(terminal=terminal(row), seconds=float(arrays[kind]['native_counts'].sum()*.002),
                               remaining_m=row['remaining_m'], terminal_contacts=row.get('terminal_contacts', []),
                               historical_reuse=bool(row.get('historical_reuse')), path=row['path']) for kind,row in rr.items()}))
    mismatch = [v['query_key'] for v in answers if v['first_action_proxy_mismatch']]
    known = all(v['complete'] for v in answers)
    return dict(queries=answers, proxy_mismatch_queries=mismatch, all_complete=known,
                evidence_state='INCOMPLETE_PANEL' if not known else ('LOCAL_PROXY_MISMATCH_OBSERVED' if mismatch else 'NO_MISMATCH_OBSERVED_IN_FIXED_PANEL'),
                next_research_decision='COMPLETE_MISSING_EVIDENCE_WITH_NEW_PROTOCOL' if not known else
                    ('LOCAL_ESTIMAND_GAP_ONLY_PROSPECTIVE_PREDICTOR_VALIDATION_STILL_NEEDED' if mismatch else
                     'FIXED_PANEL_INCONCLUSIVE_NO_AUTOMATIC_EXPANSION'),
                new_measurements=sum(not r.get('historical_reuse') for r in measured),
                historical_measurement_entries=sum(bool(r.get('historical_reuse')) for r in measured),
                predictor_trained=False, actor_updated=False, alpha_gradient_validated=False,
                independent_generalization_or_nonforgetting_proven=False, automatic_learning=False,
                inference='Matched first-hold per-query suffix replacement only; no global causal or policy-improvement claim')
