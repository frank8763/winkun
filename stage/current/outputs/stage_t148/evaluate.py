"""Deterministic, frozen four-case regression screen; these are known training seeds."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('t148_eval_train', ROOT / 'train.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
os.environ['C98_JOB'] = os.environ['TD3_JOB'] = str(ROOT)
os.environ['C22_SCREEN_ONLY'] = '0'
torch.set_num_threads(2)

def run():
    status = json.loads((ROOT / 'status.json').read_text())
    ck = ROOT / 'checkpoints' / status['checkpoint']
    checkpoint_hash = t.sha(ck)
    t.check(checkpoint_hash == status['checkpoint_sha256'], 'EVAL_CHECKPOINT_HASH_INVALID')
    saved = torch.load(ck, map_location='cpu', weights_only=False)
    core = t.load_module(t.CORE, 't148_eval_core')
    agent = t.create_agent(core)
    agent.actor.load_state_dict(saved['actor'], strict=True)
    agent.actor.eval()
    for p in agent.actor.parameters():
        p.requires_grad_(False)
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    helper = t.load_module(t.HELPER, 't148_eval_helper')
    legacy = t.load_module(t.STAGE, 't148_eval_legacy')
    for path, expected in legacy.EXPECTED.items():
        t.check(path.is_file() and legacy.sha(path) == expected, 'EVAL_DEPENDENCY_HASH_INVALID')
    runner = legacy.module(legacy.SRC, 't148_eval_runner')
    _, v, native_mod, prep, _ = runner.load_native()
    native = prep[3]
    make_base = legacy.make_short_task(runner.make_task)
    out = ROOT / ('frozen_eval_' + time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    results = []
    for seed in (2029653601, 2029654802):
        for angle in (-17.3, 18.8):
            env = renderer = None
            side = 'LEFT' if angle < 0 else 'RIGHT'
            try:
                env, renderer, obs, actual_seed = helper.make_goal_task(
                    make_base, v, native_mod, native, seed, angle, 3.0)
                env.max_steps = 14000
                initial = float(env._goal_state()[1])
                t.check(abs(initial - 3.0) < 1e-4, 'EVAL_GOAL_DISTANCE_INVALID')
                start = np.asarray(env._pose(), dtype=float)
                previous = start[:2].copy()
                travel = 0.0
                nearest = initial
                success = collision = False
                terminated = truncated = False
                trace = out / f'{seed}_{side}.jsonl'
                with trace.open('w') as f:
                    for step in range(1, 14001):
                        with torch.no_grad():
                            raw = np.asarray(agent.act(obs), dtype=np.float32)
                        t.check(raw.shape == (2,) and np.isfinite(raw).all(), 'EVAL_ACTION_INVALID')
                        action = np.asarray([(raw[0] + 1.0) / 2.0, raw[1]], dtype=np.float32)
                        obs, reward, terminated, truncated, info = native.step_checked(env, action)
                        pose = np.asarray(env._pose(), dtype=float)
                        travel += float(np.linalg.norm(pose[:2] - previous))
                        previous = pose[:2].copy()
                        distance = float(env._goal_state()[1])
                        nearest = min(nearest, distance)
                        success = bool(info.get('reach_goal', False))
                        collision = bool(info.get('collision', False))
                        f.write(json.dumps({
                            'step': step, 'pose': pose.tolist(), 'raw_action': raw.tolist(),
                            'executed_action': action.tolist(), 'remaining_m': distance,
                            'success': success, 'collision': collision,
                            'terminated': bool(terminated), 'truncated': bool(truncated),
                        }) + '\n')
                        if success or collision or terminated or truncated:
                            break
                row = {
                    'seed': seed, 'actual_seed': int(actual_seed), 'angle_deg': angle, 'side': side,
                    'initial_distance_m': initial, 'steps': step, 'success': success,
                    'collision': collision, 'remaining_m': distance, 'nearest_m': nearest,
                    'travel_m': travel, 'start_pose': start.tolist(), 'end_pose': pose.tolist(),
                    'terminated': bool(terminated), 'truncated': bool(truncated),
                    'end': 'GOAL' if success else 'COLLISION' if collision else
                           'TRUNCATED' if truncated else 'TERMINATED' if terminated else 'STEP_CAP',
                    'trace': str(trace),
                }
                results.append(row)
                print('T148_FROZEN_CASE', json.dumps(row), flush=True)
                (out / 'partial_cases.json').write_text(json.dumps(results, indent=2) + '\n')
            finally:
                if renderer is not None:
                    renderer.close()
                if env is not None:
                    env.close()
    unchanged = all(torch.equal(before[k], x) for k, x in agent.actor.state_dict().items())
    t.check(unchanged and t.sha(ck) == checkpoint_hash, 'EVAL_MODEL_CHANGED')
    report = {
        'experiment': 'STAGE_T148_FROZEN_REGRESSION_SCREEN', 'checkpoint': str(ck),
        'checkpoint_sha256': checkpoint_hash, 'training': False, 'actor_unchanged': unchanged,
        'known_training_seeds': True, 'generalization_confirmed': False,
        'automatic_promotion': False, 'cases': results,
    }
    (out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('T148_FROZEN_EVAL_COMPLETE', out, flush=True)

if __name__ == '__main__':
    run()
