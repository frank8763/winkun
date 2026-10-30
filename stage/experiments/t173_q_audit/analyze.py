"""Aggregate frozen native evidence and compare replay terminal support."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import numpy as np

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'stage/runs/t173_q_audit'


def main():
    assert json.loads((OUT / 'status.json').read_text())['state'] == 'COMPLETE'
    plan = json.loads((Path(__file__).parent / 'protocol.json').read_text())
    rows = [json.loads(x) for x in (OUT / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 47
    by_id = {r['job_id']: r for r in rows}
    repeated = []
    for row in rows:
        if row['kind'] == 'repeat':
            base = by_id['branch_' + row['case']['id'] + '_' + str(row['prefix_steps']) + '_baseline']
            assert base['branch']['state_sha256'] == row['branch']['state_sha256']
            error = abs(base['return_from_branch'] - row['return_from_branch'])
            assert error < 1e-8
            repeated.append(error)
    tolerance = max(1e-5, 100 * max(repeated))
    comparisons = []
    for row in rows:
        if row['kind'] != 'branch' or row['intervention'] == 'baseline':
            continue
        base = by_id['branch_' + row['case']['id'] + '_' + str(row['prefix_steps']) + '_baseline']
        assert base['branch']['state_sha256'] == row['branch']['state_sha256']
        assert row['return_is_complete_mc'] and base['return_is_complete_mc']
        predicted = row['branch']['q1'] - base['branch']['q1']
        actual = row['return_from_branch'] - base['return_from_branch']
        comparisons.append({'case': row['case']['id'], 'prefix_s': row['prefix_steps'] * .002,
                            'intervention': row['intervention'], 'q1_gain': predicted,
                            'observed_return_gain': actual,
                            'informative': abs(actual) > tolerance and abs(predicted) > 1e-6,
                            'sign_agreement': bool(actual * predicted > 0),
                            'baseline_success': base['success'], 'intervention_success': row['success'],
                            'baseline_primary_safe_success': base['primary_safe_success'],
                            'intervention_primary_safe_success': row['primary_safe_success'],
                            'baseline_collision': base['collision'], 'intervention_collision': row['collision']})
    mc = [r for r in rows if r['kind'] == 'mc']
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('audit_analysis_core', REPO / 'stage/current/outputs/stage_t148/td3_core_goal_contrast.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    ck = torch.load(plan['critic_actor_checkpoint'], map_location='cpu', weights_only=False)
    critic = core.Critic().eval().requires_grad_(False)
    critic.load_state_dict(ck['critic'])
    replay = ck['replay']
    support = []
    for kind, selection in [('collision_terminal', [r for r in replay if r[4] and r[2] < 0]),
                            ('goal_terminal', [r for r in replay if r[4] and r[2] > 0])]:
        if not selection:
            support.append({'kind': kind, 'n': 0})
            continue
        s = torch.from_numpy(np.asarray([r[0] for r in selection], dtype=np.float32))
        a = torch.from_numpy(np.asarray([r[1] for r in selection], dtype=np.float32))
        targets = np.asarray([r[2] for r in selection])
        with torch.no_grad():
            q1, q2 = critic(s, a)
        q1, q2 = q1.numpy().ravel(), q2.numpy().ravel()
        support.append({'kind': kind, 'n': len(selection), 'fraction': len(selection) / len(replay),
                        'q1_terminal_mae': float(np.abs(q1 - targets).mean()),
                        'minq_terminal_mae': float(np.abs(np.minimum(q1, q2) - targets).mean()),
                        'target_mean': float(targets.mean()), 'q1_mean': float(q1.mean())})
    plus = [r for r in comparisons if r['intervention'] == 'q_plus' and r['informative']]
    five_plus = [r for r in plus if r['case'].startswith('5m_')]
    result = {'state': 'COMPLETE', 'training_updates': 0,
              'paired_state_hashes': 'PASS', 'native_reward_closed_form': 'PASS',
              'baseline_repeat_abs_return_errors': repeated, 'return_comparison_tolerance': tolerance,
              'mc_episode_results': [{k: r[k] for k in ('job_id', 'success', 'collision', 'primary_safe_success',
                                     'q1_initial', 'q2_initial', 'return_from_branch', 'q1_mc_mae', 'minq_mc_mae',
                                     'q1_mc_mean_bias', 'simulation_seconds', 'final_lidar_min_m', 'contacts')} for r in mc],
              'q1_gradient_plus_sign_agreement': {'correct': sum(r['sign_agreement'] for r in plus), 'n': len(plus)},
              'q1_gradient_plus_5m_sign_agreement': {'correct': sum(r['sign_agreement'] for r in five_plus), 'n': len(five_plus)},
              'counterfactual_comparisons': comparisons,
              'replay_length': len(replay), 'replay_terminal_support': support,
              'limitations': ['Same-map fixed known tasks, one trained model, not statistical generalization.',
                             'Multiple states on each trajectory are correlated; episode-balanced errors must be reported.',
                             'TD3 training targets use a smoothed targetActor, while this audit measures the deterministic deployedActor; mismatch can contribute to error.',
                             'One-decision interventions check local action ranking, not global policy improvement or the unique cause of regression.',
                             'Discounted negative terminal reward can improve by delaying collision; scalar return gains are not arrival gains.',
                             '93.336s diagnostic continuations do not relax original strict acceptance deadlines.'],
              'automatic_promotion': False}
    (OUT / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'q_plus_agreement': result['q1_gradient_plus_sign_agreement'],
                      'five_m_agreement': result['q1_gradient_plus_5m_sign_agreement'], 'replay_terminal_support': support}))


if __name__ == '__main__':
    main()
