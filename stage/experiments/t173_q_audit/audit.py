"""Frozen on-policy value and one-decision counterfactual audit. No learning."""
from pathlib import Path
import hashlib
import importlib.util
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import sys
import time
import traceback
import fcntl
import numpy as np

CPU_POOL = [int(x) for x in os.environ.get('STAGE_CPU_POOL', ','.join(map(str, sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL'] = ','.join(map(str, CPU_POOL))
os.sched_setaffinity(0, {int(os.environ.get('STAGE_PROCESS_CPU', str(CPU_POOL[0])))})
import torch

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = REPO / 'stage/runs/t173_q_audit'
sys.path.insert(0, str(REPO / 'stage/experiments/t169_value_seed'))
from reward_contract import GAMMA, reward, checks


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(pipe, index, deadline):
    folder = OUT / 'workers' / str(index)
    folder.mkdir(parents=True, exist_ok=True)
    log = (folder / 'worker.log').open('a', buffering=1)
    sys.stdout = sys.stderr = log
    os.environ['C98_JOB'] = os.environ['TD3_JOB'] = str(folder)
    os.environ['C22_SCREEN_ONLY'] = '0'
    env = renderer = None
    try:
        import mujoco
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        plan = json.loads((HERE / 'protocol.json').read_text())
        core = load(REPO / 'stage/current/outputs/stage_t148/td3_core_goal_contrast.py', 'q_audit_core')
        actor, teacher, critic = core.Actor().eval(), core.Actor().eval(), core.Critic().eval()
        ck = torch.load(plan['critic_actor_checkpoint'], map_location='cpu', weights_only=False)
        actor.load_state_dict(ck['actor'])
        critic.load_state_dict(ck['critic'])
        del ck
        ck = torch.load(plan['incumbent_checkpoint'], map_location='cpu', weights_only=False)
        teacher.load_state_dict(ck['actor'])
        del ck
        for net in (actor, teacher, critic):
            net.requires_grad_(False)
        frozen = [{k: v.clone() for k, v in net.state_dict().items()} for net in (actor, teacher, critic)]
        t = load(REPO / 'stage/current/outputs/stage_t148/train.py', 'q_audit_source')
        helper = t.load_module(t.HELPER, 'q_audit_helper')
        legacy = t.load_module(t.STAGE, 'q_audit_legacy')
        for p, h in legacy.EXPECTED.items():
            assert sha(p) == h, p
        runner = legacy.module(legacy.SRC, 'q_audit_runner')
        _, v, mod, prep, _ = runner.load_native()
        native = prep[3]
        make = legacy.make_short_task(runner.make_task)
        checks()
        while True:
            job = pipe.recv()
            if job is None:
                break
            if env is not None:
                env.close()
            case = job['case']
            env, renderer, obs, actual = helper.make_goal_task(make, v, mod, native, case['seed'], case['angle_deg'], case['distance_m'])
            if renderer is not None:
                renderer.close()
                renderer = None
            env.max_steps = plan['diagnostic_step_cap']
            goal = env.goal_xy.copy()
            assert not env._collision() and abs(float(env.model.opt.timestep) - .002) < 1e-10
            blocks, states, actions, predictions = [], [], [], []
            terminal = trunc = False
            steps = 0
            branch = None
            while not (terminal or trunc):
                assert time.monotonic() < deadline, 'Diagnostic wall budget'
                x = torch.from_numpy(np.asarray(obs, dtype=np.float32).copy()).reshape(1, -1)
                with torch.no_grad():
                    raw = actor(x).numpy()[0]
                at_branch = steps == job['prefix_steps']
                if at_branch:
                    a = torch.tensor(raw.reshape(1, -1), requires_grad=True)
                    q1, q2 = critic(x, a)
                    gradient = torch.autograd.grad(q1.sum(), a)[0].numpy()[0]
                    direction = gradient / max(1e-12, float(np.linalg.norm(gradient)))
                    if job['intervention'] == 'q_plus':
                        raw = np.clip(raw + .25 * direction, -1, 1).astype(np.float32)
                    elif job['intervention'] == 'q_minus':
                        raw = np.clip(raw - .25 * direction, -1, 1).astype(np.float32)
                    elif job['intervention'] == 'T159':
                        with torch.no_grad():
                            raw = teacher(x).numpy()[0]
                    else:
                        assert job['intervention'] == 'baseline'
                    integration = np.zeros(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
                    mujoco.mj_getState(env.model, env.data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
                    fingerprint = hashlib.sha256(integration.tobytes() + np.asarray(obs, dtype=np.float32).tobytes() + goal.tobytes()).hexdigest()
                    branch = {'state_sha256': fingerprint, 'observation': obs.tolist(),
                              'action': raw.tolist(), 'q1_gradient': gradient.tolist(),
                              'distance_m': float(env._goal_state()[1])}
                with torch.no_grad():
                    p1, p2 = critic(x, torch.from_numpy(raw.reshape(1, -1)))
                if at_branch:
                    branch.update(q1=float(p1), q2=float(p2))
                state_before = np.asarray(obs, dtype=np.float32).copy()
                raw_before = raw.copy()
                total = 0.
                mapped = np.array([(raw[0] + 1) / 2, raw[1]], dtype=np.float32)
                for n in range(1, 26):
                    distance = float(env._goal_state()[1])
                    time0 = float(env.data.time)
                    obs, _, terminal, trunc, info = native.step_checked(env, mapped)
                    assert abs(float(env.data.time) - time0 - .002) < 1e-7
                    obs = np.asarray(obs, dtype=np.float32)
                    total += GAMMA ** (n - 1) * reward(distance, float(env._goal_state()[1]), info['reach_goal'], info['collision'], terminal)
                    steps += 1
                    if terminal or trunc:
                        break
                if steps > job['prefix_steps']:
                    blocks.append((total, GAMMA ** n, n))
                    if job['kind'] == 'mc':
                        states.append(state_before)
                        actions.append(raw_before)
                        predictions.append([float(p1), float(p2)])
            assert branch is not None and np.array_equal(goal, env.goal_xy)
            # No bootstrap is inserted into observed MC labels. Truncations remain censored.
            observed = 0.
            returns = []
            for r, g, _ in reversed(blocks):
                observed = r + g * observed
                returns.append(observed)
            returns.reverse()
            result = {'job_id': job['id'], 'kind': job['kind'], 'case': case,
                      'prefix_steps': job['prefix_steps'], 'intervention': job['intervention'],
                      'actual_seed': int(actual), 'steps': steps, 'branch': branch,
                      'terminal': bool(terminal), 'truncated': bool(trunc),
                      'success': bool(info['reach_goal']), 'collision': bool(info['collision']),
                      'return_from_branch': observed, 'return_is_complete_mc': bool(terminal),
                      'primary_safe_success': bool(info['reach_goal'] and not info['collision'] and steps <= case['step_cap']),
                      'remaining_m': float(env._goal_state()[1]), 'simulation_seconds': float(env.data.time),
                      'final_lidar_min_m': float(env._lidar48().min() * 6),
                      'contacts': [{'g1': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                    'g2': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
                                    'distance': float(c.dist)} for c in env.data.contact[:env.data.ncon]]}
            if terminal:
                native_count = sum(b[2] for b in blocks)
                discount = GAMMA ** native_count
                # Independent closed-form check of the declared terminal-consistent shaped objective.
                base = -.00002 * (1 - discount) / (1 - GAMMA)
                base += GAMMA ** (native_count - 1) * (10 if info['reach_goal'] else -10 if info['collision'] else 0)
                expected = base + .3 * branch['distance_m']
                assert abs(expected - observed) < 1e-7, (expected, observed)
            if job['kind'] == 'mc':
                q = np.asarray(predictions)
                labels = np.asarray(returns)
                trace = OUT / (job['id'] + '.npz')
                np.savez_compressed(trace, states=np.asarray(states), actions=np.asarray(actions),
                                    block_rewards=np.array([b[0] for b in blocks]),
                                    discounts=np.array([b[1] for b in blocks]),
                                    native_counts=np.array([b[2] for b in blocks]),
                                    q_predictions=q, observed_returns=labels)
                result.update(trace=str(trace.relative_to(REPO)), sampled_transitions=len(states),
                              q1_initial=float(q[0, 0]), q2_initial=float(q[0, 1]),
                              q1_mc_mae=float(np.abs(q[:, 0] - labels).mean()) if terminal else None,
                              minq_mc_mae=float(np.abs(q.min(axis=1) - labels).mean()) if terminal else None,
                              q1_mc_mean_bias=float((q[:, 0] - labels).mean()) if terminal else None)
            for net, before in zip((actor, teacher, critic), frozen):
                assert all(torch.equal(before[k], v) for k, v in net.state_dict().items())
            pipe.send(result)
    except (EOFError, SystemExit):
        pass
    except BaseException:
        try:
            pipe.send({'error': traceback.format_exc()})
        except OSError:
            pass
        raise
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()
        pipe.close()
        log.close()


def main():
    lock = (OUT / 'process.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'status.json').exists(), 'No duplicate/restart'
    plan = json.loads((HERE / 'protocol.json').read_text())
    for p, h in plan['input_hashes'].items():
        assert sha(p) == h, p
    assert sum(p.stat().st_size for p in (REPO / 'stage/runs').rglob('*') if p.is_file()) + 64 * 1024**2 < 1024**3
    jobs = list(plan['jobs'])
    ctx = mp.get_context('spawn')
    processes, pipes = [], []
    results = []
    start = time.monotonic()
    deadline = start + plan['wall_seconds']
    try:
        for index in range(8):
            a, b = ctx.Pipe()
            process = ctx.Process(target=worker, args=(b, index, deadline))
            os.environ['STAGE_PROCESS_CPU'] = str(CPU_POOL[(index + 1) % len(CPU_POOL)])
            process.start()
            os.environ['STAGE_PROCESS_CPU'] = str(CPU_POOL[0])
            b.close()
            processes.append(process)
            pipes.append(a)
        save(OUT / 'workers.json', {'parent_pid': os.getpid(), 'pids': [p.pid for p in processes]})
        active = set(pipes)
        for pipe in pipes:
            pipe.send(jobs.pop(0))
        while active:
            assert time.monotonic() < deadline
            ready = wait(list(active), timeout=1)
            for pipe in ready:
                row = pipe.recv()
                assert 'error' not in row, row
                results.append(row)
                with (OUT / 'results.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                frames = sum(r['steps'] for r in results)
                assert frames <= plan['maximum_physics_steps']
                save(OUT / 'status.json', {'state': 'RUNNING', 'completed_jobs': len(results),
                                          'total_jobs': len(plan['jobs']), 'completed_physics_steps': frames,
                                          'gradient_updates': 0, 'elapsed_s': time.monotonic() - start})
                if jobs:
                    pipe.send(jobs.pop(0))
                else:
                    pipe.send(None)
                    active.remove(pipe)
        assert len(results) == len(plan['jobs'])
        save(OUT / 'status.json', {'state': 'COMPLETE', 'completed_jobs': len(results),
                                  'completed_physics_steps': sum(r['steps'] for r in results),
                                  'gradient_updates': 0, 'elapsed_s': time.monotonic() - start})
        print(json.dumps({'state': 'COMPLETE', 'jobs': len(results)}), flush=True)
    finally:
        for pipe in pipes:
            try:
                pipe.send(None)
            except (OSError, EOFError):
                pass
        for process in processes:
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
        for pipe in pipes:
            pipe.close()


if __name__ == '__main__':
    main()
