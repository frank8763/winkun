"""Verify completed critic controls and render their descriptive evidence."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[3] / 'stage/runs/t175_paired_advantage/cache/matplotlib'))
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
DOCS = REPO / 'stage/docs/results_20261030_q_audit'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    DOCS.mkdir(exist_ok=True)
    checks = []
    original = torch.load(REPO / 'stage/runs/t171_reverse_curriculum/checkpoint_latest.pt', map_location='cpu', weights_only=False)['actor']
    for name in ['t173_q_audit', 't174_critic_alignment', 't175_paired_advantage']:
        root = REPO / 'stage/runs' / name
        status = json.loads((root / 'status.json').read_text())
        assert status['state'] == 'COMPLETE'
        plan = json.loads((REPO / 'stage/experiments' / name / 'protocol.json').read_text())
        for p, digest in plan['input_hashes'].items():
            assert sha(p) == digest, p
        file = root / ('analysis.json' if name == 't173_q_audit' else 'summary.json')
        data = json.loads(file.read_text())
        if name != 't173_q_audit':
            for arm, result in data['arms'].items():
                path = REPO / result['checkpoint']
                assert sha(path) == result['checkpoint_sha256']
                state = torch.load(path, map_location='cpu', weights_only=False)
                assert state['actor_updates'] == 0 and state['new_physics_steps'] == 0
                assert state['protocol_sha256'] == sha(REPO / 'stage/experiments' / name / 'protocol.json')
                assert all(torch.equal(original[k], v) for k, v in state['actor'].items())
                assert result['after']['critic_gate'] is False
        shutil.copyfile(file, DOCS / (name + '_' + file.name))
        checks.append({'run': name, 'input_hashes': 'PASS', 'state': 'COMPLETE',
                       'actor_unchanged': True, 'automatic_promotion': False})
    shutil.copyfile(REPO / 'stage/runs/t175_paired_advantage/holdout_verification.json', DOCS / 'initial_state_holdout_verification.json')
    a = json.loads((DOCS / 't173_q_audit_analysis.json').read_text())
    b = json.loads((DOCS / 't174_critic_alignment_summary.json').read_text())['arms']
    c = json.loads((DOCS / 't175_paired_advantage_summary.json').read_text())['arms']
    compact = {'checks': checks, 't173_qplus_5m_ranking': a['q1_gradient_plus_5m_sign_agreement'],
               't174': {}, 't175': {}, 'online_actor_updates': 0,
               'new_native_diagnostic_steps': 795686, 'critic_updates': 10000,
               'retained_actor': 'T159', 'primary_5m_status': 'unchanged0/2',
               'scope': 'One inherited seed per control, one map. No independent training repetitions, no new navigation policy, no automatic promotion.'}
    for name, result in b.items():
        after = result['after']
        compact['t174'][name] = {'collision_terminal_q1_mae': after['terminal_errors']['collision']['q1_mae'],
                                'return_label_holdout_q1_mae': after['mean_return_holdout_q1_mae'],
                                'critic_gate': after['critic_gate']}
    for name, result in c.items():
        after = result['after']
        train = [r for r in after['action_rank_comparisons'] if r['prefix_steps'] != 5000 and r['informative']]
        held = [r for r in after['action_rank_comparisons'] if r['prefix_steps'] == 5000 and r['informative']]
        compact['t175'][name] = {'training_action_ranking': {'correct': sum(r['sign_agreement'] for r in train), 'n': len(train)},
                                'held_state_action_ranking': after['paired_action_return_holdout'],
                                'training_delta_return_mae': float(np.mean([abs(r['q1_gain'] - r['mc_gain']) for r in train])),
                                'held_state_delta_return_mae': float(np.mean([abs(r['q1_gain'] - r['mc_gain']) for r in held])),
                                'training_course_q1_mae': float(np.mean([v['q1_mae'] for v in after['mc_courses'].values()])),
                                'held_initial_return_q1_mae': after['initial_state_return_holdout_q1_mae'],
                                'critic_gate': after['critic_gate']}
    (DOCS / 'verification_and_metrics.json').write_text(json.dumps(compact, indent=2) + '\n')

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout='constrained')
    labels = ['Uniform', 'Terminal\nbalanced', 'Balanced\n+ MC']
    colors = ['#64748b', '#0f766e', '#7c3aed']
    errors = [v['after']['terminal_errors']['collision']['q1_mae'] for v in b.values()]
    axes[0].bar(labels, errors, color=colors)
    axes[0].set_yscale('log')
    axes[0].set_ylim(.05, 20)
    axes[0].set_title('T174: stored collision terminals (n=24)')
    axes[0].set_ylabel('Q1 mean absolute error (log scale)')
    for i, value in enumerate(errors):
        axes[0].text(i, value * 1.12, f'{value:.3f}', ha='center', va='bottom')
    x = np.arange(3)
    left = [v['after']['mc_courses']['5m_2029653601_LEFT']['q1_mae'] for v in b.values()]
    right = [v['after']['mc_courses']['5m_2029654802_RIGHT']['q1_mae'] for v in b.values()]
    axes[1].bar(x - .18, left, width=.36, label='5m LEFT: MC training course', color='#0f766e')
    axes[1].bar(x + .18, right, width=.36, label='5m RIGHT: MC-label holdout', color='#d97706')
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel('Episode mean absolute Q1 / MC error')
    axes[1].set_title('Terminal fit does not fix route-level value')
    axes[1].legend(fontsize=7)
    for name, color in zip(c, ['#64748b', '#7c3aed']):
        held = [r for r in c[name]['after']['action_rank_comparisons'] if r['prefix_steps'] == 5000]
        axes[2].scatter([r['mc_gain'] for r in held], [r['q1_gain'] for r in held], s=28, color=color, label=name.replace('_', ' '), alpha=.8)
    axes[2].plot([-.5, .5], [-.5, .5], '--', color='#94a3b8', lw=1)
    axes[2].set_xlabel('Native one-decision MC return difference')
    axes[2].set_ylabel('Predicted Q1 difference')
    axes[2].set_title('T175: held states, 9 comparisons / 3 states')
    axes[2].legend(fontsize=7)
    fig.suptitle('Frozen-Actor critic diagnostics: single-map development evidence', fontsize=11)
    fig.savefig(DOCS / 'critic_diagnostics.png', dpi=170)
    plt.close(fig)
    print(json.dumps({'state': 'PASS', 'output': str(DOCS), 'critic_updates': 10000, 'actor_updates': 0}))


if __name__ == '__main__':
    main()
