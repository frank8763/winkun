"""Register a new curriculum hypothesis only after completed T259 evidence."""
from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
from course import LEVELS, contracts

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t263_adaptive_goal_course'


def main():
    assert not OUT.exists(), 'No duplicate T263'
    prior = ROOT/'stage/runs/t259_closed_loop_recovery'
    summary = json.loads((prior/'summary.json').read_text())
    assert summary['state'] == 'COMPLETE' and summary['all_workers_stopped']
    assert summary['old_regression_n'] == summary['old_regression_safe'] == 275
    assert summary['new_development_n'] == 54 and not summary['new_distance_gate']
    assert summary['heldout_n'] == 0, 'Do not reuse consumed final six-metre holdout'
    contracts()
    plan = copy.deepcopy(json.loads((ROOT/'stage/experiments/t259_closed_loop_recovery/protocol.json').read_text()))
    for key in ['engineering_recovery_of', 'combined_original_deadline_epoch',
                'failed_T258_output_bytes', 'prelaunch_budget_enforcement',
                'bridge_cases', 'bridge_rounds', 'bridge_distance_m', 'historical_training_rows']:
        plan.pop(key, None)
    bases = []
    for level, distance in enumerate(LEVELS):
        for side in ['LEFT', 'RIGHT']:
            ref = next(j for j in plan['development_jobs'] if j['case']['side'] == side)
            case = copy.deepcopy(ref['case'])
            case.update(id=f'{distance:.2f}m_{side}', distance_m=distance,
                        step_cap=math.ceil(distance/3*14000))
            name = f'coursebase_{level}_{side}'
            bases.append(dict(id=name, source_reset_id=name, group='course_base',
                case=case, delta_x_m=0., delta_y_m=0., delta_yaw_deg=0.,
                seed=0, mode='deterministic', course_level=level,
                snapshot_destination=str(OUT/'tasks'/f'course_{level}_{side}.npz')))
    plan.update(experiment='T263', course_levels_m=list(LEVELS), course_bases=bases,
        initial_course_levels=dict(LEFT=0, RIGHT=0), rounds=6, draws_per_reset=4,
        max_commands=3200, max_physics=90000000, wall_seconds=5400,
        final_reserve_seconds=1000, final_bytes=64*1024**2,
        training_episode_bytes=384*1024, save_full_training_states=True,
        historical_training_logs=[str(ROOT/'stage/runs/t257_distance_policy_gradient/training.jsonl'),
                                  str(prior/'training.jsonl')],
        contract='New controlled training-goal curriculum hypothesis, not resumed T259. '
        'Requires T259 complete/all workers stopped/old275all safe/new54notallpass/heldout16unconsumed. '
        'Fresh SAME T255 accepted5m function and fresh Adam; SAME120D hidden-reset singleActor, '
        'seed2026110101, six rounds x four tasks x four draws=96max, independent exact regeneration, '
        'four Actor attempts/shadow, conditional AR, PCGrad, learningrates/reward/softretention100, '
        'same fixed12 native old-task acceptance and trustbounds. Only training-goal schedule changes: '
        'LEFT and RIGHT begin5.25; fixed levels5.25/5.5/5.75/6. After each completed update round, '
        'advance one level only with >=2/8safe exploratory draws on that side, actual retained update, '
        'and current-goal deterministic center safe; zero-safe plus failed center retreats one level. '
        'Otherwise hold; no distance/seed/threshold search. Every episode one real fixed finalgoal '
        'from original factory pose/heading; no within-episode goal clipping/waypoints/planner/router/teacher. '
        'All course base/18yawzero positions perlevel plus54dev/16heldout locked zero-physics before learning; '
        'same-side course bases must match scene/pose/fullnativeintegration, initial120D fields excluding '
        'currentgoal48/51/52 match within2e-6, and actual goalradius matches requested distance1e-10. '
        'Actual6m centers always separately reported; course success not6m success. Final lastmodel '
        '275old+54sixm; onlyallpass16unconsumedFRESHholdout and78independent reload exact; then7m2centers '
        'frozen only, noautomatic7mtraining. Allguard/replay/geometry/validation overhead counted. '
        'Atmost96 training episodes and same90M totalphysics ceiling asT259, actualphysics/wallnotguaranteedmatched; '
        'do notclaim algorithm superiority. Save full120Dstates/nextstates onlytrainingwithin64MiB; '
        'history exact overlaps identified by allarray/integration hashes acrossT257/T259 regardlessjobIDs. '
        'Stop old8centerloss/two0update rounds/budgets; interrupted Actor+Adam round fully rolledback. '
        'No new Critic/BC/feature/memory/reward/noise/Phase. Global1GiB/free40GiB, no deleting evidence or oldrunrestart.',
        references=plan['references']+['https://proceedings.mlr.press/v80/florensa18a.html',
                                      'https://arxiv.org/abs/1910.07224'])
    paths = list(HERE.glob('*.py')) + [prior/'summary.json', prior/'training.jsonl']
    plan['input_hashes'].update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    for path, expected in plan['input_hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    used = sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used + plan['final_bytes'] < plan['global_bytes'], 'No storage reservation available'
    assert shutil.disk_usage(ROOT).free >= plan['min_free_bytes']
    data = json.dumps(plan, ensure_ascii=False, indent=2)+'\n'
    OUT.mkdir()
    (HERE/'protocol.json').write_text(data)
    (OUT/'protocol.json').write_text(data)
    print(json.dumps(dict(state='REGISTERED', original_T259_gate=False,
        max_training_episodes=96, runs_MiB=used/1024**2)), flush=True)


if __name__ == '__main__':
    main()
