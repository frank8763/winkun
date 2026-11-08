"""Independent zero-step audit of static post-step endpoint contacts.

No Stage environment, Actor, learner or producer module is imported. Loading an
endpoint and calling mj_forward is NOT a replay of its native contact event.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import signal
import time
import traceback

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'stage/runs/t430_full_body_contact'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def disk_bytes(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


def audit():
    start = time.time()
    assert not (OUT / 'audit.json').exists()
    assert not (OUT / 'audit_error.json').exists()
    p = read(OUT / 'protocol.json')
    deadline = min(start + 120., float(p['deadline_all']))

    def guard(*_):
        assert time.time() < deadline, 'Independent audit deadline'
        assert disk_bytes(OUT) < 4 * 2**20, 'Audit output budget'
        assert disk_bytes(ROOT / 'stage/runs') < 2**30, 'Global logical budget'
        assert shutil.disk_usage(ROOT).free >= 40 * 2**30, 'Free disk budget'

    def timeout(*_):
        raise TimeoutError('Independent endpoint audit deadline')

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, max(.01, deadline - time.time()))
    for path, expected in p['inputs'].items():
        assert sha(path) == expected, ('Input SHA', path)
    summary = read(OUT / 'summary.json')
    assert summary['state'] == 'COMPLETE'
    assert summary.get('physics', summary.get('physics_steps')) == 0
    assert summary.get('Adam', summary.get('Adam_calls')) == 0
    main_pid = summary.get('pid', summary.get('main_pid'))
    assert type(main_pid) is int and main_pid != os.getpid()
    assert not Path('/proc', str(main_pid)).exists(), 'Producer still alive'

    # Banning all exposed integration entry points makes an accidental dynamic
    # call explicit; mj_forward alone computes endpoint kinematics/contacts.
    def forbidden(*_args, **_kwargs):
        raise AssertionError('Independent audit forbids native integration')
    mujoco.mj_step = forbidden
    mujoco.mj_step1 = forbidden
    mujoco.mj_step2 = forbidden
    model = mujoco.MjModel.from_binary_path(str(p['model_path']))
    geometry = read(p['geometry_path'])
    geoms = geometry['geoms']
    assert model.ngeom == len(geoms) == 53
    tolerance = float(p['geometry_json_tolerance'])
    assert tolerance == 1e-12
    robot = set()
    for g in geoms:
        i = g['id']
        assert int(model.geom_type[i]) == g['type']
        assert int(model.geom_bodyid[i]) == g['body']
        assert int(model.geom_contype[i]) == g['contype']
        assert int(model.geom_conaffinity[i]) == g['conaffinity']
        assert np.allclose(model.geom_pos[i], g['local_pos'], atol=tolerance, rtol=0)
        assert np.allclose(model.geom_size[i], g['size'], atol=tolerance, rtol=0)
        assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) == g['name']
        if g['robot']:
            robot.add(i)
    assert robot
    base_body = min(int(model.geom_bodyid[i]) for i in robot)

    def is_floor(i):
        geom = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[i])) or ''
        return (int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
                or any(word in (geom + ' ' + body).lower() for word in ['floor', 'ground', 'plane']))

    producer = read(OUT / 'contact_results.json')
    rows = producer['rows'] if isinstance(producer, dict) else producer
    by_label = {row['label']: row for row in rows}
    assert len(by_label) == len(rows) == len(p['snapshots'])
    assert set(by_label) == {s['label'] for s in p['snapshots']}
    results = []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    assert mujoco.mj_stateSize(model, spec) == 203
    for snapshot in p['snapshots']:
        guard()
        assert sha(snapshot['path']) == snapshot['sha256']
        with np.load(snapshot['path'], allow_pickle=False) as raw:
            state = np.asarray(raw['final_native_integration']).copy()
            original_pose = np.asarray(raw['next_poses'][-1]).copy()
            assert bool(raw['end_flags'][-1, 0]) == snapshot['expected_original_collision']
        assert state.shape == (203,) and state.dtype == np.float64
        assert np.isfinite(state).all()
        data = mujoco.MjData(model)
        mujoco.mj_setState(model, data, state, spec)
        mujoco.mj_forward(model, data)
        contacts = []
        obstacles = []
        for i in range(data.ncon):
            contact = data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            entry = dict(geom1=g1, geom2=g2,
                         name1=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1),
                         name2=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2),
                         distance_m=float(contact.dist))
            contacts.append(entry)
            one_robot = (g1 in robot) != (g2 in robot)
            other = g2 if g1 in robot else g1
            if one_robot and not is_floor(other):
                obstacles.append(entry)
        matrix = data.xmat[base_body].reshape(3, 3)
        pose = [float(data.xpos[base_body, 0]), float(data.xpos[base_body, 1]),
                float(math.atan2(matrix[1, 0], matrix[0, 0]))]
        mujoco.mj_setState(model, data, state, spec)
        recovered = np.empty(203, np.float64)
        mujoco.mj_getState(model, data, recovered, spec)
        assert np.array_equal(recovered, state), 'Integration state roundtrip'
        row = by_label[snapshot['label']]
        assert row['source_sha256'] == snapshot['sha256']
        assert row['integration_sha256'] == hashlib.sha256(state.tobytes()).hexdigest()
        assert row['contacts'] == contacts, ('All endpoint contacts', snapshot['label'])
        assert row['obstacle_contacts'] == obstacles
        reconstructed_collision = any(c['distance_m'] <= .002 for c in obstacles)
        assert row['reconstructed_obstacle_collision'] == reconstructed_collision
        assert row['integration_roundtrip_exact'] is True
        # Matrix-based independent yaw vs producer's free-joint quaternion
        # expression can differ by a few ULP; this is representation equality,
        # not a relaxed physical contact or success threshold.
        assert np.allclose(np.asarray(row['pose_xy_yaw']), np.asarray(pose), atol=1e-10, rtol=0)
        assert np.allclose(original_pose, np.asarray(pose), atol=1e-10, rtol=0)
        results.append(dict(label=snapshot['label'], contacts=len(contacts),
                            obstacle_contacts=obstacles,
                            expected_original_collision=snapshot['expected_original_collision'],
                            reconstructed_obstacle_collision=reconstructed_collision,
                            equality_to_original_collision_label=reconstructed_collision == snapshot['expected_original_collision'],
                            integration_roundtrip_exact=True))
    guard()
    result = dict(state='PASS', pid=os.getpid(), elapsed_s=time.time() - start,
                  physics=0, Adam=0, new_Actor_updates=0,
                  static_geometry_fields_checked=53,
                  old_geometry_json_rounding_tolerance=tolerance,
                  full_historical_model_SHA_equivalence_claimed=False,
                  current_model_sha256=sha(p['model_path']),
                  producer_absent_pid=main_pid, all_producer_contacts_recomputed=True,
                  dynamic_step_entry_points_banned=True, rows=results,
                  limitations=['Static post-step endpoints only; not native trajectory or contact-event replay.',
                               'Original collision labels may disagree; such disagreements are reported, not hidden.',
                               'The old geometry JSON does not identify all historical model dynamics.'])
    save_new(OUT / 'audit.json', result)
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(state='PASS', endpoint_snapshots=len(results), physics=0, Adam=0)))


if __name__ == '__main__':
    try:
        audit()
    except BaseException:
        error = traceback.format_exc()
        if OUT.exists() and not (OUT / 'audit_error.json').exists():
            save_new(OUT / 'audit_error.json', dict(state='FAILED', pid=os.getpid(),
                                                   error=error, physics=0, Adam=0))
        raise
