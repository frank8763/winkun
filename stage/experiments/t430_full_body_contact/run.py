"""Bounded, zero-integration contact reconstruction from saved Stage endpoints.

No environment, Actor, learner or renderer is imported or constructed. This
does not replay navigation or reproduce a solver's pre-integration contacts.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t430_full_body_contact'
DOC = ROOT / 'stage/docs/contact_20261108'
MODEL = Path('/home/linghu/mkmini_stage_daad_g2e_overnight_gate_v1/run/models/reference.mjb')
XML = Path('/home/linghu/mkmini_stage_daad_native_layout_recovery_v4_2/run/reference.xml')
GEOMETRY = ROOT / 'stage/runs/t326_gap_geometry/geometry.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def used(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def verify(p):
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for path, value in p['inputs'].items():
        assert sha(path) == value, path
    assert used(OUT) <= p['max_bytes'] and used(ROOT / 'stage/runs') <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40 * 2**30
    assert time.time() < p['deadline_all']


def prepare():
    assert not OUT.exists()
    assert sha(MODEL) == '5faeeb1db0d212f97f855f4e17e9f33e6f9cdcdcfce80281674dfa87ee76ad76'
    assert sha(XML) == 'a7eec29e9fa5c4db88e3b41f93fb9e37e936ddd4532fdfd4b29e16eb6c8f7acf'
    prior = ROOT / 'stage/runs/t429_feedback_recovery'
    q = read(prior / 'protocol.json')
    target = next(v for v in q['recoveries'] if v['job_id'] == 'dev_task6_21')
    rows = [json.loads(line) for line in (prior / 'native/cases.jsonl').read_text().splitlines()]
    failed = next(r for r in rows if r['job_id'] == target['job_id'])
    assert read(prior / 'native/audit.json')['state'] == 'PASS'
    reference = dict(path=target['reference_path'], sha256=target['reference_sha256'])
    snapshots = [dict(label=label, path=row['path'], sha256=row['sha256'], expected_original_collision=collision)
                 for label, row, collision in [
                     ('student_original', target['student_row'], True),
                     ('neural_teacher_recovery', target['failed_mixed_row'], True),
                     ('fixed_feedback_recovery', failed, True),
                     ('successful_reference', reference, False)]]
    inputs = {str(f) for f in HERE.glob('*.py')} | {
        str(MODEL), str(XML), str(GEOMETRY),
        '/home/linghu/stage_daad_g4h_multibranch_night_v1/runner/native_env.py',
        '/home/linghu/mkmini_stage_daad_g2c_exact_factory_entry_v1/run/report.json',
        str(prior / 'protocol.json'),
        str(prior / 'native/cases.jsonl'), str(prior / 'native/audit.json'),
        str(DOC / 'PLAN.md'), str(DOC / 'PRIMARY_SOURCES.md')}
    inputs |= {item['path'] for item in snapshots}
    now = time.time()
    deadline = datetime(2026, 11, 8, 21, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    assert now + 540 < deadline
    cap = 4 * 2**20
    assert used(ROOT / 'stage/runs') + cap < 2**30
    p = dict(id='T430_ZERO_STEP_FULL_BODY_CONTACT_RECONSTRUCTION',
             model_path=str(MODEL), xml_path=str(XML), geometry_path=str(GEOMETRY),
             model_sha256=sha(MODEL), snapshots=snapshots, inputs={f: sha(f) for f in sorted(inputs)},
             max_physics=0, max_optimizer_calls=0, max_bytes=cap,
             main_seconds=120, audit_seconds=120, backup_seconds=180, source_seconds=120,
             deadline_all=deadline, geometry_json_tolerance=1e-12,
             metadata_tolerance_scope='Old JSON coordinate rounding only; not a physics/contact tolerance.',
             collision_distance_threshold_m=.002,
             native_steps_prohibited=True, no_Actor=True,
             hypothesis='Static endpoint contact may involve protruding wheel geoms despite positive chassis-only gap.',
             no_native_navigation_or_recovery_attempt=True)
    OUT.mkdir()
    save(HERE / 'protocol.json', p)
    save(OUT / 'protocol.json', p)
    save(DOC / 'BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', physics=0, Actor_Adam=0,
         root_bytes=cap, global_full_reserved=used(ROOT / 'stage/runs') + cap,
         deadline_all='2026-11-08 21:00 Asia/Shanghai', max_main_audit_backup_source_s=540))
    print('T430_REGISTERED_ZERO_PHYSICS')


def preflight():
    p = read(OUT / 'protocol.json')
    verify(p)
    assert not (OUT / 'preflight.json').exists()
    import numpy as np
    import mujoco as mj
    attempts = []
    def prohibited(*args, **kwargs):
        attempts.append(1)
        raise AssertionError('T430 forbids physical integration')
    for name in ('mj_step', 'mj_step1', 'mj_step2'):
        setattr(mj, name, prohibited)
        try:
            getattr(mj, name)(None, None)
        except AssertionError:
            pass
        else:
            raise AssertionError('Missing zero-step guard')
    assert len(attempts) == 3
    for row in p['snapshots']:
        assert sha(row['path']) == row['sha256']
        with np.load(row['path'], allow_pickle=False) as a:
            assert a['final_native_integration'].shape == (203,)
            assert bool(a['end_flags'][-1, 0]) == row['expected_original_collision']
    save(OUT / 'preflight.json', dict(state='PASS', native_steps=0, Actor_Adam=0,
         actual_numpy_payloads_verified=4, synthetic_step_guard_rejections=3,
         model_not_constructed=True, no_environment_or_Torch_import=True))
    print('T430_PREFLIGHT_PASS_ZERO_PHYSICS')


def launch():
    p = read(OUT / 'protocol.json')
    verify(p)
    assert read(OUT / 'preflight.json')['state'] == 'PASS'
    assert not (OUT / 'launch.json').exists()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    receipt = read(ROOT / 'stage/runs/t224_archive/latest_source_sync_receipt.json')
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    assert receipt['state'] == 'REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree'] == tree
    assert time.time() + 540 < p['deadline_all']
    with (DOC / 'CONSOLE.log').open('xb') as log:
        child = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=2',
            str(int(p['deadline_all'] - time.time() - 5)), 'taskset', '-c', '0',
            str(ROOT / 'run_stage.sh'), str(HERE / 'driver.py')], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT / 'launch.json', dict(pid=child.pid, epoch=time.time(), source_remote=receipt['remote_commit'],
         root_tree=tree, protocol_sha256=sha(OUT / 'protocol.json'), zero_physics=True,
         deadline_all=p['deadline_all']))
    print('T430_ACTUALLY_LAUNCHED_ZERO_PHYSICS', child.pid)


def main():
    import numpy as np
    import mujoco as mj
    p = read(OUT / 'protocol.json')
    verify(p)
    assert not (OUT / 'summary.json').exists() and not (OUT / 'main_started.json').exists()
    started = time.time()
    save(OUT / 'main_started.json', dict(pid=os.getpid(), epoch=started, physics=0))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T430 main bound')))
    signal.alarm(p['main_seconds'])
    attempts = []
    def prohibited(*args, **kwargs):
        attempts.append(1)
        raise AssertionError('T430 forbids physical integration')
    for name in ('mj_step', 'mj_step1', 'mj_step2'):
        setattr(mj, name, prohibited)
    try:
        geometry = read(p['geometry_path'])
        m = mj.MjModel.from_binary_path(p['model_path'])
        d = mj.MjData(m)
        assert len(geometry['geoms']) == m.ngeom == 53
        differences = []
        for old in geometry['geoms']:
            i = old['id']
            assert (mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or None) == old['name']
            for key, field in [('type', 'geom_type'), ('body', 'geom_bodyid'),
                               ('contype', 'geom_contype'), ('conaffinity', 'geom_conaffinity')]:
                assert int(getattr(m, field)[i]) == old[key], (i, key)
            for key, field in [('local_pos', 'geom_pos'), ('size', 'geom_size')]:
                error = float(np.max(np.abs(getattr(m, field)[i] - np.asarray(old[key]))))
                differences.append(error)
                assert error <= p['geometry_json_tolerance'], (i, key, error)
        assert mj.mj_stateSize(m, mj.mjtState.mjSTATE_INTEGRATION) == 203
        robots = {g['id'] for g in geometry['geoms'] if g['robot']}
        results = []
        for snapshot in p['snapshots']:
            verify(p)
            with np.load(snapshot['path'], allow_pickle=False) as a:
                state = a['final_native_integration'].copy()
                expected_pose = a['next_poses'][-1].copy()
            mj.mj_setState(m, d, state, mj.mjtState.mjSTATE_INTEGRATION)
            mj.mj_forward(m, d)
            # Preserve original integration/warmstart; contacts are derived
            # at the recorded endpoint, not a new integrated transition.
            mj.mj_setState(m, d, state, mj.mjtState.mjSTATE_INTEGRATION)
            actual = np.empty(203)
            mj.mj_getState(m, d, actual, mj.mjtState.mjSTATE_INTEGRATION)
            assert np.array_equal(actual, state)
            contacts, obstacles = [], []
            for contact in d.contact[:d.ncon]:
                g1, g2 = int(contact.geom1), int(contact.geom2)
                n1 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g1)
                n2 = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, g2)
                value = dict(geom1=g1, geom2=g2, name1=n1, name2=n2, distance_m=float(contact.dist))
                contacts.append(value)
                if (g1 in robots) == (g2 in robots):
                    continue
                other = g2 if g1 in robots else g1
                body = int(m.geom_bodyid[other])
                names = ((mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, other) or '') + ' ' +
                         (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, body) or '')).lower()
                if any(token in names for token in ('floor', 'ground', 'plane')):
                    continue
                obstacles.append(value)
            q = d.qpos[3:7]
            yaw = np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2]**2 + q[3]**2))
            pose = np.array([d.qpos[0], d.qpos[1], yaw])
            assert np.max(np.abs(pose - expected_pose)) < 1e-10
            results.append(dict(label=snapshot['label'], source_sha256=snapshot['sha256'],
                integration_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                contacts=contacts, obstacle_contacts=obstacles,
                reconstructed_obstacle_collision=any(c['distance_m'] <= .002 for c in obstacles),
                pose_xy_yaw=pose.tolist(), integration_roundtrip_exact=True,
                original_collision=snapshot['expected_original_collision']))
        assert not attempts
        save(OUT / 'contact_results.json', dict(rows=results, physics=0, Actor_Adam=0,
            step_function_attempts=0, model_sha256=p['model_sha256'],
            maximum_old_JSON_geometry_rounding_error=max(differences),
            geometry_scope='Static collider configuration; no assertion of full runtime dynamics identity.',
            original_solver_contact_timing_not_replayed=True))
        summary = dict(state='COMPLETE', pid=os.getpid(), physics=0, Adam=0, Actor_updates=0,
            elapsed_s=time.time() - started, model_constructions=1, static_endpoint_queries=len(results),
            all_workers_stopped=True, workers_spawned=0, no_navigation_or_recovery_attempt=True)
        save(OUT / 'summary.json', summary)
        print(json.dumps(summary))
    except BaseException:
        import traceback
        save(OUT / 'error.json', dict(error=traceback.format_exc(), physics=0,
             step_function_attempts=len(attempts), original_files_unchanged=True))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ('prepare', 'preflight', 'launch', 'main')
    globals()[sys.argv[1]]()
