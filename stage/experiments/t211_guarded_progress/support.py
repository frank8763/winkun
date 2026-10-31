"""Stage-only full-parameter PPO pilot with a fresh, separate state baseline."""
import os
os.sched_setaffinity(0, {int(os.environ.get('STAGE_PROCESS_CPU', '0'))})
from pathlib import Path
import sys, time, json, traceback, math
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t211_guarded_progress'
sys.path.insert(0, str(ROOT / 'stage/experiments/t203_native_exploration'))
import runtime as R
C, np, torch = R.C, R.np, R.torch


class Value(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(113, 128), torch.nn.Tanh(),
                                       torch.nn.Linear(128, 128), torch.nn.Tanh(), torch.nn.Linear(128, 1))
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def policy(state):
    p = R.Policy()
    p.load_state_dict(state)
    p.requires_grad_(True)
    p.logstd.requires_grad_(False)  # T203 fixed native-domain sigma, not an unused trainable scalar.
    return p


def monte_carlo(rewards, counts):
    """Complete finite-horizon episodes only: no bootstrap across a real timeout."""
    result = np.empty(len(rewards), dtype=np.float64)
    acc = 0.
    for i in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[i]) + R.GAMMA ** int(counts[i]) * acc
        result[i] = acc
    return result


def training_rewards(rewards, counts, terminal_distance, coefficient):
    result = np.array(rewards, dtype=np.float64, copy=True)
    # A changed task objective, NOT a policy-invariant potential-shaping claim.
    result[-1] -= R.GAMMA ** (int(counts[-1]) - 1) * coefficient * terminal_distance
    return result


def used(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def worker(pipe, index, deadline):
    folder = OUT / 'workers' / str(index)
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'worker.log').open('a', buffering=1)
    sys.stdout = sys.stderr = log
    env = None
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        native, create = C.native_setup(folder)
        while True:
            command = pipe.recv()
            if command is None:
                break
            kind, job = command
            if kind == 'make_grid':
                env, obs, _ = R.task_env(create, job)
                valid = not env._collision()
                path = OUT / 'tasks' / (job['id'] + '.npz')
                path.parent.mkdir(exist_ok=True)
                if valid:
                    assert not path.exists()
                    C.write_snapshot(path, C.capture(env, obs))
                pipe.send({'valid': valid, 'snapshot': str(path) if valid else None,
                           'physics_steps': 0, 'spec': job})
                env.close(); env = None
                continue
            assert kind in ('rollout', 'replay')
            assert C.sha(job['weights']) == job['weights_sha256']
            p = policy(torch.load(job['weights'], map_location='cpu', weights_only=False)['policy']).eval().requires_grad_(False)
            env, obs, _ = R.task_env(create, job)
            assert not env._collision()
            original_model, original_goal = C.model_hash(env), env.goal_xy.copy()
            steps, cap = 0, job['case']['step_cap']
            rng = np.random.default_rng(job['seed'])
            info, records = {}, []
            nearest = float(env._goal_state()[1])
            initial_distance = nearest
            term = trunc = False
            if kind == 'replay':
                assert C.sha(job['trace']) == job['trace_sha256']
                with np.load(job['trace']) as z:
                    a = {k: z[k].copy() for k in z.files}
                state_error = action_error = logp_error = 0.
                for i in range(len(a['actions'])):
                    if time.time() >= deadline:
                        raise TimeoutError('Replay deadline; evidence not compacted')
                    state = C.observation(obs, 1 - steps / cap)
                    action, _, lp = R.draw(p, state, job['mode'], rng)
                    state_error = max(state_error, float(np.abs(state - a['states'][i]).max()))
                    action_error = max(action_error, float(np.abs(action - a['actions'][i]).max()))
                    logp_error = max(logp_error, abs(lp - float(a['logp'][i])))
                    for _ in range(int(a['native_counts'][i])):
                        obs, _, term, trunc, info = native.step_checked(env, action)
                        steps += 1
                    state_error = max(state_error, float(np.abs(C.observation(obs, 1 - steps/cap) - a['next_states'][i]).max()))
                integration_hash = R.arrays_hash(R.integration(env))
                assert state_error == 0 and action_error == 0 and logp_error == 0
                assert integration_hash == job['integration_sha256']
                row = {'id': job['id'], 'verified': True, 'physics_steps': steps,
                       'state_error': state_error, 'action_error': action_error, 'logp_error': logp_error}
            else:
                while not (term or trunc) and steps < cap and time.time() < deadline:
                    state = C.observation(obs, 1 - steps/cap)
                    action, mean, lp = R.draw(p, state, job['mode'], rng)
                    before, d0 = steps, float(env._goal_state()[1])
                    pose = np.asarray(env._pose()[:3])
                    for _ in range(min(100, cap-steps)):
                        obs, _, term, trunc, info = native.step_checked(env, action)
                        steps += 1
                        nearest = min(nearest, float(env._goal_state()[1]))
                        if term or trunc:
                            break
                    n, d1 = steps-before, float(env._goal_state()[1])
                    discount = R.GAMMA ** n
                    terminal = term or trunc
                    event = float(info['reach_goal']) - float(info['collision']) - .5*float(trunc)
                    reward = (-.000002*(1-discount)/(1-R.GAMMA) + .03*d0
                              - discount*.03*d1*(not terminal)
                              + (R.GAMMA**(n-1)*event if terminal else 0))
                    records.append((state, action, mean, lp, n, C.observation(obs, 1-steps/cap),
                                    pose, np.asarray(env._pose()[:3]), [info['collision'], info['reach_goal'], trunc], reward))
                complete = bool(term or trunc)
                assert records
                arrays = {k: np.asarray([r[i] for r in records]) for i, k in enumerate(
                    ['states', 'actions', 'means', 'logp', 'native_counts', 'next_states', 'poses', 'next_poses', 'end_flags', 'rewards'])}
                arrays['native_counts'] = arrays['native_counts'].astype(np.uint8)
                arrays['end_flags'] = arrays['end_flags'].astype(np.uint8)
                arrays['goal_xy'] = original_goal
                arrays['remaining_m'] = np.asarray(float(env._goal_state()[1]))
                path = OUT / 'episodes' / (job['id'] + '.npz')
                path.parent.mkdir(exist_ok=True)
                assert not path.exists()
                # All failures retained. Evaluation omits redundant observations from the outset.
                if job['role'] != 'training':
                    arrays = {k: v for k, v in arrays.items() if k not in ('states', 'next_states', 'means', 'logp')}
                np.savez_compressed(path, **arrays)
                assert path.stat().st_size < 256*1024
                row = {k: job[k] for k in ('id', 'arm', 'round', 'role', 'mode', 'seed', 'weights', 'weights_sha256', 'case', 'exact_snapshot')}
                row.update(complete=complete, physics_steps=steps, decisions=len(records),
                           success=bool(info.get('reach_goal')), collision=bool(info.get('collision')), timeout=bool(trunc),
                           remaining_m=float(env._goal_state()[1]), nearest_m=nearest, initial_distance_m=initial_distance,
                           environment_return=float(monte_carlo([r[-1] for r in records], [r[4] for r in records])[0]) if complete else None,
                           trace=str(path), trace_sha256=C.sha(path), integration_sha256=R.arrays_hash(R.integration(env)))
            assert original_model == C.model_hash(env) and np.array_equal(original_goal, env.goal_xy)
            env.close(); env = None
            pipe.send(row)
    except (EOFError, SystemExit):
        pass
    except BaseException:
        pipe.send({'error': traceback.format_exc()})
        raise
    finally:
        if env is not None:
            env.close()
        pipe.close(); log.close()
