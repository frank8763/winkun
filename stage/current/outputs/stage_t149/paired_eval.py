"""Frozen paired development evaluation; never imports or updates Phase planners."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'stage_t148'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)

def signature(items):
    h = hashlib.sha256()
    for name, value in items:
        a = np.asarray(value)
        if np.issubdtype(a.dtype, np.floating):
            a = np.round(a.astype('<f8'), 8)
        h.update(name.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    out = plan_path.parent
    lock = (out / 'process.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'summary.json').exists():
        raise RuntimeError('ALREADY_COMPLETED')
    if (out / 'cases.jsonl').exists():
        raise RuntimeError('PARTIAL_RUN_EXISTS_USE_NEW_OUTPUT')
    for p, sha in plan['source_hashes'].items():
        assert digest(p) == sha, ('SOURCE_CHANGED', p)
    torch.set_num_threads(1)
    os.environ['C98_JOB'] = os.environ['TD3_JOB'] = str(out)
    os.environ['C22_SCREEN_ONLY'] = '0'
    spec = importlib.util.spec_from_file_location('t149_train', SOURCE / 'train.py')
    t = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t)
    helper = t.load_module(t.HELPER, 't149_helper')
    legacy = t.load_module(t.STAGE, 't149_legacy')
    for path, sha in legacy.EXPECTED.items():
        assert digest(path) == sha
    runner = legacy.module(legacy.SRC, 't149_runner')
    _, v, native_mod, prep, _ = runner.load_native()
    native = prep[3]
    make_base = legacy.make_short_task(runner.make_task)
    core = t.load_module(t.CORE, 't149_core')
    agents, before = {}, {}
    for name, model in plan['models'].items():
        assert digest(model['checkpoint']) == model['sha256']
        ck = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        agent = t.create_agent(core)
        agent.actor.load_state_dict(ck['actor'], strict=True)
        agent.actor.eval()
        agent.actor.requires_grad_(False)
        agents[name] = agent
        before[name] = {k: x.clone() for k, x in agent.actor.state_dict().items()}
        del ck
    rows = []
    pair_signatures = {}
    deadline = time.monotonic() + plan['wall_limit_seconds']
    for case_idx, case in enumerate(plan['cases']):
        # Alternate evaluation order, without changing case membership.
        order = list(agents) if case_idx % 2 == 0 else list(reversed(agents))
        for name in order:
            env = renderer = None
            try:
                env, renderer, obs, actual = helper.make_goal_task(
                    make_base, v, native_mod, native, case['seed'], case['angle_deg'], 3.0)
                env.max_steps = plan['step_cap']
                obs = np.asarray(obs, dtype=np.float32)
                pose0 = np.asarray(env._pose(), dtype=float)
                distance0 = float(env._goal_state()[1])
                assert abs(distance0 - 3.0) < 1e-4 and not env._collision()
                scene = signature([(k, getattr(env.model, k)) for k in
                    ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_bodyid',
                     'geom_contype', 'geom_conaffinity', 'body_pos', 'body_quat')])
                task_sig = signature([('scene', np.frombuffer(bytes.fromhex(scene), dtype='u1')),
                    ('qpos', env.data.qpos), ('qvel', env.data.qvel), ('ctrl', env.data.ctrl),
                    ('goal', env.goal_xy), ('obs', obs)])
                if case['id'] in pair_signatures:
                    assert pair_signatures[case['id']] == task_sig, 'PAIRED_INITIAL_STATE_MISMATCH'
                else:
                    pair_signatures[case['id']] = task_sig
                model = agents[name]
                previous = pose0[:2].copy()
                start_time = float(env.data.time)
                nearest, travel, dt = distance0, 0.0, None
                success = collision = terminated = truncated = False
                milestones = {}
                trace = out / f'{case["id"]}_{name}.jsonl'
                with trace.open('w') as f:
                    for step in range(1, plan['step_cap'] + 1):
                        if time.monotonic() >= deadline:
                            raise TimeoutError('T149_WALL_LIMIT_NO_PARTIAL_PROMOTION')
                        raw = np.asarray(model.act(obs), dtype=np.float32)
                        assert raw.shape == (2,) and np.isfinite(raw).all() and np.max(np.abs(raw)) <= 1.00001
                        action = np.array([(raw[0] + 1) / 2, raw[1]], dtype=np.float32)
                        time0 = float(env.data.time)
                        obs, reward, terminated, truncated, info = native.step_checked(env, action)
                        elapsed_step = float(env.data.time) - time0
                        if dt is None:
                            dt = elapsed_step
                        assert elapsed_step > 0 and abs(elapsed_step - dt) < 1e-7
                        obs = np.asarray(obs, dtype=np.float32)
                        assert obs.shape == (112,) and np.isfinite(obs).all()
                        pose = np.asarray(env._pose(), dtype=float)
                        travel += float(np.linalg.norm(pose[:2] - previous))
                        previous = pose[:2].copy()
                        distance = float(env._goal_state()[1])
                        bearing = math.degrees(float(env._goal_state()[2]))
                        nearest = min(nearest, distance)
                        success = bool(info.get('reach_goal', False))
                        collision = bool(info.get('collision', False))
                        point = {'step': step, 'remaining_m': distance, 'bearing_deg': bearing,
                                 'raw_action': raw.tolist(), 'pose': pose.tolist()}
                        for threshold in (1.2, 0.8, 0.5, 0.3):
                            if distance <= threshold and str(threshold) not in milestones:
                                milestones[str(threshold)] = point
                        if step == 1 or step % 20 == 0 or success or collision or terminated or truncated:
                            f.write(json.dumps({**point, 'simulation_time_s': float(env.data.time),
                                'success': success, 'collision': collision,
                                'terminated': bool(terminated), 'truncated': bool(truncated),
                                'native_reward': float(reward)}) + '\n')
                        if success or collision or terminated or truncated:
                            break
                row = {**case, 'model': name, 'actual_seed': int(actual),
                    'scene_signature': scene, 'task_signature': task_sig,
                    'start_pose': pose0.tolist(), 'goal_xy': np.asarray(env.goal_xy).tolist(),
                    'initial_distance_m': distance0, 'steps': step, 'success': success,
                    'safe_success': success and not collision, 'collision': collision,
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'remaining_m': distance, 'nearest_m': nearest, 'final_bearing_deg': bearing,
                    'travel_m': travel, 'step_simulation_s': dt,
                    'simulation_duration_s': float(env.data.time) - start_time,
                    'goal_radius_m': float(env.success_radius), 'milestones': milestones,
                    'end': 'COLLISION' if collision else 'GOAL' if success else
                           'TRUNCATED' if truncated else 'TERMINATED' if terminated else 'STEP_CAP',
                    'trace_file': str(trace)}
                rows.append(row)
                with (out / 'cases.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                save(out / 'progress.json', {'completed': len(rows), 'total': len(plan['cases']) * len(agents),
                    'last': {k: row[k] for k in ['id', 'model', 'end', 'remaining_m']}})
                print('T149_CASE', json.dumps({k: row[k] for k in ['id','model','end','steps','remaining_m','nearest_m','step_simulation_s']}), flush=True)
            finally:
                if renderer is not None:
                    renderer.close()
                if env is not None:
                    env.close()
    unchanged = {name: all(torch.equal(before[name][k], x) for k, x in agent.actor.state_dict().items())
                 for name, agent in agents.items()}
    assert all(unchanged.values())
    for model in plan['models'].values():
        assert digest(model['checkpoint']) == model['sha256']
    result = {'experiment': plan['experiment'], 'plan_sha256': digest(plan_path),
        'actor_unchanged': unchanged, 'training': False, 'automatic_promotion': False,
        'training_runs_per_model': 1, 'generalization_claim': False,
        'unique_scene_signatures': len({r['scene_signature'] for r in rows}),
        'unique_task_signatures': len({r['task_signature'] for r in rows}), 'cases': rows}
    save(out / 'summary.json', result)
    print('T149_COMPLETE', out / 'summary.json', flush=True)

if __name__ == '__main__':
    main()
