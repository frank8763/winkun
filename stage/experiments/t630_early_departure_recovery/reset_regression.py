"""Exercise the actual failed reset closure, pose reader and history witness.

The environment seam supplies arrays and observations only: no model, native
integration, Actor, optimizer or CUDA is initialized. Original sources stay
immutable. A seam result does not claim a successful native reset or navigation.
"""
from pathlib import Path
import ast
import collections
import copy
import hashlib
import json
import math
import sys
import types
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT / 'stage/docs/day_20261118/t630'
FIELD = ROOT / 'stage/experiments/t502_fullfield_probe/t502_runtime.py'
POSE = ROOT / 'stage/runtime/mk-mini/training/e2e_phase10_lidar_v1/phase10_scene_e2e_env.py'
WITNESS = ROOT / 'stage/experiments/t295_hindsight_support/run.py'
CONTRACT = WITNESS.with_name('t295_contract.py')


def definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace[name]


def fixture(job, fixed=False):
    ns = dict(np=np, math=math, collections=collections, copy=copy)
    fields = definition(CONTRACT, 'fields', ns)
    Witness = definition(WITNESS, 'Witness', ns)
    pose = definition(POSE, '_pose', ns)

    class Env:
        _pose = pose

        def __init__(self):
            self.data = types.SimpleNamespace(qpos=np.array([0., 0., .1, 1., 0., 0., 0.]),
                                             qvel=np.zeros(6), ctrl=np.zeros(2), time=0.)
            self.model = object()
            self.goal_xy = np.array([1., 0.])
            self.history = collections.deque(maxlen=4)
            self.prev_action = np.zeros(2, np.float32)

        def _history_frame(self, lidar, distance, angle):
            f = np.zeros(12, np.float32)
            f[8] = np.clip(distance / 10, 0, 2)
            f[9] = np.clip(angle / math.pi, -1, 1)
            return f

        def _observation(self, update_history=False):
            if not self.history:
                _, distance, angle = fields(self._pose(), self.goal_xy)
                for _ in range(4):
                    self.history.append(self._history_frame(None, distance, angle))
            return np.concatenate(list(self.history))

        def _collision(self):
            return False

    env = Env()
    witness = Witness(Env)
    original_model = sys.modules.get('mujoco')
    fake_native = types.SimpleNamespace(mj_forward=lambda model, data: None)
    sys.modules['mujoco'] = fake_native
    target = types.SimpleNamespace()
    integration = lambda e: np.r_[e.data.qpos, e.data.qvel, e.data.ctrl, e.data.time]
    deps = types.SimpleNamespace(
        BASE_SETUP=lambda folder, p: (None, None, None, witness, None),
        BASE_TASK=lambda create, j: (env, env._observation(), 0),
        E=types.SimpleNamespace(N=types.SimpleNamespace(
            science=lambda p: p, K=types.SimpleNamespace(R=target))),
        np=np, math=math, copy=copy)
    target.integration = integration
    try:
        if fixed:
            from reset_contract import field_setup
            field_setup(deps, None, {})
        else:
            scope = vars(deps).copy()
            setup = definition(FIELD, 'field_setup', scope)
            setup(None, {})
        target.task_env(None, copy.deepcopy(job))
        hp, hg, ht = witness.get(env)
        return dict(qpos=env.data.qpos.copy(), history_pose=hp, history_goal=hg,
                    history_time=ht, observation=env._observation(),
                    integration=integration(env))
    finally:
        witness.close()
        if original_model is None:
            del sys.modules['mujoco']
        else:
            sys.modules['mujoco'] = original_model


def main():
    p = json.loads((ROOT/'stage/runs/t629_offline_night_relay/protocol.json').read_text())
    jobs = [j for chunk in p['neighborhood_chunks'] for j in chunk]
    red = []
    for job in jobs:
        try:
            fixture(job)
        except AssertionError:
            red.append(job['id'])
    failed = 't502_field_sector1_offset2_T629_dx0.0_yaw2.0'
    assert failed in red, 'Original failure no longer reproduced'
    report = dict(state='RED_REPRODUCED',original_rejected_ids=red,
                  actual_original_field_closure_pose_and_witness=True,
                  seam_only_not_native_initialization=True,
                  native_steps=0, optimizer_updates=0)
    if '--fixed' in sys.argv:
        from reset_contract import check_history
        unchanged = 0
        for job in jobs:
            good = fixture(job, fixed=True)
            if job['id'] not in red:
                old = fixture(job)
                assert all(np.array_equal(good[k], old[k]) for k in good)
                unchanged += 1
        pose0 = np.array([1., 2., math.pi+.02])
        hp = np.tile([1.,2.,-math.pi+.02], (4,1))
        goal = np.array([3.,4.]); hg=np.tile(goal,(4,1)); ht=np.zeros(4)
        check_history(hp,hg,ht,pose0,goal)
        rejected = []
        for name, index, delta in [('x',0,.01),('y',1,.01),('yaw',2,.01)]:
            wrong=hp.copy();wrong[0,index]+=delta
            try:check_history(wrong,hg,ht,pose0,goal)
            except AssertionError:rejected.append(name)
        for name,wrongg,wrongt in [('goal',hg+.01,ht),('time',hg,ht+.01),
                                  ('nan',hg*np.nan,ht)]:
            try:check_history(hp,wrongg,wrongt,pose0,goal)
            except AssertionError:rejected.append(name)
        assert len(rejected)==6
        report.update(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER', fixed_jobs=len(jobs),
                      canonical_initial_arrays_bitwise_unchanged=unchanged,
                      true_position_yaw_goal_time_nan_errors_rejected=rejected)
    report['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in [FIELD,POSE,WITNESS,CONTRACT,Path(__file__),
                                       Path(__file__).with_name('reset_contract.py')]
                             if f.exists()}
    DOC.mkdir(parents=True,exist_ok=True)
    name='RESET_PREFLIGHT.json' if '--fixed' in sys.argv else 'RESET_RED.json'
    (DOC/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
