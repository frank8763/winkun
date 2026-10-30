"""Read-only Stage observation/time audit; 64 native steps, no learning."""
from pathlib import Path
import importlib.util
import inspect
import json
import math
import os
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent

def main():
    assert not (ROOT / 'audit.json').exists(), 'Keep prior audit immutable'
    os.environ['C98_JOB'] = os.environ['TD3_JOB'] = str(ROOT)
    os.environ['C22_SCREEN_ONLY'] = '0'
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('t162_env_source', ROOT.parent / 'stage_t148/train.py')
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    helper = t.load_module(t.HELPER, 't162_helper')
    legacy = t.load_module(t.STAGE, 't162_legacy')
    for path, sha in legacy.EXPECTED.items():
        assert t.sha(path) == sha
    runner = legacy.module(legacy.SRC, 't162_runner')
    _, v, mod, prep, _ = runner.load_native(); native = prep[3]
    core = t.load_module(t.CORE, 't162_core')
    actor_path = ROOT.parent / 'stage_t159/consolidated_actor.pt'
    actor_sha = t.sha(actor_path)
    assert actor_sha == 'd0065843f6766407038bae679b32177b20991a429010bb3cb83872413f8ea570'
    agent = t.create_agent(core)
    agent.actor.load_state_dict(torch.load(actor_path, map_location='cpu', weights_only=False)['actor'])
    agent.actor.eval().requires_grad_(False)
    before = {k: x.clone() for k, x in agent.actor.state_dict().items()}
    env = renderer = None
    try:
        env, renderer, obs, actual = helper.make_goal_task(legacy.make_short_task(runner.make_task),
            v, mod, native, 2029653601, -17.3, 3.)
        sources = {}
        for name in ('_observation', '_history_frame', '_goal_state', '_pose', '_lidar48', '_clearance', '_policy_to_commands'):
            fn = getattr(env, name); path = Path(inspect.getsourcefile(fn))
            sources[name] = {'path': str(path), 'sha256': t.sha(path),
                'start_line': inspect.getsourcelines(fn)[1], 'source': inspect.getsource(fn)}
        fn = t._t59_external_clearance_reward
        sources['training_clearance_reward'] = {'path': str(t.__file__), 'sha256': t.sha(t.__file__),
            'start_line': inspect.getsourcelines(fn)[1], 'source': inspect.getsource(fn)}
        checks = []; timings = []; history_error = []; current_error = []
        for i in range(64):
            old = np.asarray(obs).copy(); start = float(env.data.time)
            raw = agent.act(obs)
            obs, _, terminal, truncated, info = native.step_checked(env,
                np.array([(raw[0]+1)/2, raw[1]], dtype=np.float32))
            timings.append(float(env.data.time)-start)
            obs = np.asarray(obs)
            history_error.append(float(np.max(np.abs(obs[64:100]-old[76:112]))))
            lidar = env._lidar48(); gf, dist, ang = env._goal_state()
            frame = env._history_frame(lidar, dist, ang)
            current = np.concatenate([lidar, gf, env.prev_action, env._clearance(lidar), [float(np.mean(lidar))]])
            current_error.append(float(np.max(np.abs(obs[:64]-current))))
            checks.append(bool(np.allclose(obs[100:112], frame, atol=1e-7)))
            assert not terminal and not truncated and not info['collision']
        assert max(history_error) == 0 and max(current_error) < 1e-7 and all(checks)
        dt = float(np.mean(timings)); assert abs(dt-.002) < 1e-10
        assert all(torch.equal(before[k], x) for k, x in agent.actor.state_dict().items())
        assert t.sha(actor_path) == actor_sha
        report = {'experiment': 'STAGE_T162_OBSERVATION_AND_TIME_CONTRACT_AUDIT',
            'training': False, 'native_steps': 64, 'actual_seed': int(actual),
            'actor_path': str(actor_path), 'actor_sha256': actor_sha,
            'sources': sources, 'physics_dt_s': dt, 'actor_decision_hz': 1/dt,
            'history_frames': 4, 'history_includes_current_frame': True,
            'oldest_history_age_s': 3*dt, 'history_shift_max_error': max(history_error),
            'current_observation_reconstruction_max_error': max(current_error),
            'latest_history_matches_current': all(checks),
            'gamma': .99, 'discount_efold_s': -dt/math.log(.99),
            'discount_half_life_s': -dt*math.log(2)/math.log(.99),
            'actor_and_checkpoint_unchanged': True,
            'inference_observation': {'lidar': '48 simulated rays; current relative scan',
                'goal': 'body-frame final-goal vector/distance/bearing from simulator pose',
                'motion': 'body velocity and yaw rate from simulator qpos/qvel',
                'clearance': 'summaries of the same rays, not obstacle-coordinate inputs',
                'history': 'four 12-value summaries, newest is current; explicit action columns masked in T159'},
            'deployment_gap': 'Pose/velocity are exact simulation state, not yet noisy odometry or measured real sensors. Mapless is not localization-free.',
            'training_privilege': 'T59 reward accesses simulator geometry distances; this is training reward, not a policy observation. Declare in experiments.',
            'scope': 'Observation construction and 64-step runtime verified; not a navigation evaluation or proof that short memory causes failures.'}
        (ROOT/'audit.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k != 'sources'}, indent=2))
    finally:
        if renderer is not None: renderer.close()
        if env is not None: env.close()

if __name__ == '__main__': main()
