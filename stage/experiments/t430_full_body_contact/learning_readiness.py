"""POSTHOC NumPy-only QA of already audited recovery learning support.

No Actor construction, forward/backward, optimizer or simulator is imported.
This does not register or start training, nor produce new successful labels.
"""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / 'stage/docs/contact_20261108'
INDEX = ROOT / 'stage/docs/feedback_20261108/RECOVERY_SUPPORT.json'
OWN = ROOT / 'stage/runs/t427_fixed_representation/learn/cases.jsonl'
DEST = DOC / 'LEARNING_READINESS_REVIEWED.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def key(state, hidden):
    h = hashlib.sha256()
    # Explicit dimensions/dtypes matter: numerical proximity is not equality.
    for a in (state, hidden):
        a = np.asarray(a)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.digest()


def residual(mean, actions):
    mean = np.asarray(mean, np.float64)
    actions = np.asarray(actions, np.float64)
    assert mean.shape == actions.shape and mean.shape[-1] == 2
    return np.where(actions == 1, np.maximum(0, 1 - mean)**2,
                    np.where(actions == -1, np.maximum(0, mean + 1)**2,
                             (mean - actions)**2))


def summaries(a, first):
    location = np.asarray(a['locations'], np.float64)[first:]
    action = np.asarray(a['actions'], np.float64)[first:]
    error = np.abs(np.clip(location, -1, 1).astype(np.float32) - action)
    loss = residual(location, action)
    return dict(decisions=len(action), executed_MAE_by_drive_steer=error.mean(0).tolist(),
                executed_max_error_by_drive_steer=error.max(0).tolist(),
                existing_one_sided_loss=float(loss.mean()),
                existing_loss_by_drive_steer=loss.mean(0).tolist(),
                teacher_exact_boundary_fraction_by_drive_steer=(np.abs(action) == 1).mean(0).tolist(),
                executed_error_p50_p90_p99_by_drive_steer=np.quantile(error, [.5, .9, .99], axis=0).tolist())


def analyze():
    started = time.time()
    index = read(INDEX)
    assert index['state'] == 'VERIFIED_RECOVERY_LABEL_INDEX_NO_TRAINING'
    successes = index['successes']
    assert len(successes) == 6 and len({r['job_id'] for r in successes}) == 6
    policy_hashes = {r['policy_hash'] for r in successes}
    assert len(policy_hashes) == 1
    own_rows = [json.loads(line) for line in OWN.read_text().splitlines()]
    anchors = [r for r in own_rows if r['full'] and r['success'] and not r['collision']]
    assert len(anchors) == 59 and {r['policy_hash'] for r in anchors} == policy_hashes
    # This tests literal contradictory targets at identical (input,hidden),
    # not gradient conflict or feasibility of a restricted parameterization.
    own = {}
    own_steps = 0
    inputs = {str(INDEX): sha(INDEX), str(OWN): sha(OWN)}
    for r in anchors:
        assert sha(r['path']) == r['sha256']
        inputs[r['path']] = r['sha256']
        with np.load(r['path'], allow_pickle=False) as a:
            assert not a['ownership'].any() and not a['stochastic'].any()
            assert np.array_equal(np.clip(a['locations'], -1, 1).astype(np.float32), a['actions'])
            for i in range(len(a['states'])):
                k = key(a['states'][i], a['hidden_before'][i])
                own.setdefault(k, []).append((r['job_id'], a['locations'][i].copy()))
                own_steps += 1
    rows = []
    all_targets = {}
    pair_conflicts, anchor_conflicts = [], []
    pair_comparisons = 0
    anchor_comparisons = 0
    for r in successes:
        assert sha(r['path']) == r['sha256']
        assert sha(r['source_audit_path']) == r['source_audit_sha256']
        assert read(r['source_audit_path'])['state'] == 'PASS'
        inputs[r['path']] = r['sha256']
        inputs[r['source_audit_path']] = r['source_audit_sha256']
        with np.load(r['path'], allow_pickle=False) as a:
            first = r['first_feedback_or_teacher_hold']
            assert not a['ownership'][:first].any() and a['ownership'][first:].all()
            assert not a['stochastic'].any()
            assert np.array_equal(a['end_flags'][-1], [0, 1, 0]) and not a['end_flags'][:-1].any()
            assert np.array_equal(np.clip(a['locations'][:first], -1, 1).astype(np.float32), a['actions'][:first])
            n = len(a['actions'])
            first_window_end = min(first + 10, n)
            suffix = summaries(a, first)
            early_a = {'locations': a['locations'][:first_window_end], 'actions': a['actions'][:first_window_end]}
            early = summaries(early_a, first)
            matched = 0
            for i in range(first, n):
                k = key(a['states'][i], a['hidden_before'][i])
                action = a['actions'][i]
                for old_id, mean in own.get(k, []):
                    matched += 1
                    anchor_comparisons += 1
                    value = residual(mean[None], action[None])
                    if value.max() > 1e-20:
                        anchor_conflicts.append(dict(recovery=r['job_id'], hold=i, anchor=old_id,
                                                     minimum_preserved_mean_loss=float(value.mean())))
                for other in all_targets.get(k, []):
                    pair_comparisons += 1
                    if not np.array_equal(other['action'], action):
                        pair_conflicts.append(dict(first=other['id'], second=r['job_id'], hold=i))
                all_targets.setdefault(k, []).append(dict(id=r['job_id'], action=action.copy()))
            rows.append(dict(job_id=r['job_id'], source_path=r['path'], source_sha256=r['sha256'],
                behavior=r['behavior'], first_hold=first, actual_handoff_seconds=first * .2,
                prefix_decisions_excluded_from_teacher_labels=first, suffix=suffix,
                first_at_most_2s_after_handoff=early,
                first_student_executed_action=np.clip(a['locations'][first], -1, 1).astype(np.float32).tolist(),
                first_actual_teacher_action=a['actions'][first].tolist(),
                exact_input_hidden_anchor_matches=matched, causal_student_locations_previously_audited=True))
    total = sum(r['suffix']['decisions'] for r in rows)
    return dict(state='POSTHOC_VERIFIED_SUPPORT_QA_NO_LEARNING', source_sha256=sha(__file__),
        elapsed_s=time.time() - started, inputs=inputs, new_native_steps=0, Actor_forwards=0,
        Actor_updates=0, optimizer_calls=0, backward_calls=0, successful_recovery_cases=6,
        teacher_label_decisions=total, original_student_success_anchor_cases=59,
        original_student_success_anchor_decisions=own_steps, rows=rows,
        mean_loss_weighting_note='Existing correction objective averages each complete recovery, then averages cases equally.',
        unweighted_case_average_existing_loss=float(np.mean([r['suffix']['existing_one_sided_loss'] for r in rows])),
        exact_recovery_label_pair_conflicts=pair_conflicts,
        exact_recovery_vs_preserved_location_conflicts=anchor_conflicts,
        exact_recovery_label_pair_comparisons=pair_comparisons,
        exact_recovery_vs_anchor_comparisons=anchor_comparisons,
        comparison_note='Zero conflicts with zero matching keys is an empty comparison, not evidence against interference.',
        no_new_model_or_labels_or_navigation_acceptance=True,
        limitations=['Recorded causal student locations are authenticated by prior full-policy audits, not recomputed here.',
          'Exact-key tests do not measure near-state ambiguity, gradient interference, feature rank or capacity.',
          'Zero exact contradictions would not prove that 66 parameters can fit the constraints.',
          'Frozen-student errors on teacher-visited trajectories do not predict autonomous navigation success.',
          '59 full anchors are 51 six-metre cases plus 8 old centres, not all 271 old successful tasks.',
          'No learning mechanism, optimizer budget, or start time is registered by this report.'])


if __name__ == '__main__':
    assert not DEST.exists()
    result = analyze()
    assert result['elapsed_s'] < 120
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['state', 'successful_recovery_cases', 'teacher_label_decisions',
          'original_student_success_anchor_cases', 'unweighted_case_average_existing_loss']}, indent=2))
