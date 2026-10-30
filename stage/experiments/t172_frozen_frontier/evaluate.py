"""Frozen matched reset audit of T159 versus T171. No gradients or selection."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import time
import numpy as np

CPU_POOL = [int(x) for x in os.environ.get('STAGE_CPU_POOL', ','.join(map(str, sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL'] = ','.join(map(str, CPU_POOL))
os.sched_setaffinity(0, {int(os.environ.get('STAGE_PROCESS_CPU', str(CPU_POOL[0])))})
import torch

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = REPO / 'stage/runs/t172_frozen_frontier'
sys.path.insert(0, str(REPO / 'stage/experiments/t171_reverse_curriculum'))
from worker import run


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    temp = p.with_suffix('.partial')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, p)


def receive(pipe):
    assert pipe.poll(60), 'Native worker timeout'
    row = pipe.recv()
    assert 'error' not in row, row
    return row


def main():
    lock = (OUT / 'process.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'status.json').exists(), 'Never duplicate/restart a completed audit'
    plan = json.loads((HERE / 'protocol.json').read_text())
    for p, digest in plan['input_hashes'].items():
        assert sha(p) == digest, p
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('frozen_frontier_core', REPO / 'stage/current/outputs/stage_t148/td3_core_goal_contrast.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    rows = []
    initial_tasks = {}
    elapsed_start = time.monotonic()
    deadline = elapsed_start + plan['wall_seconds']
    frames = 0
    save('status.json', {'state': 'RUNNING', 'gradient_updates': 0, 'completed_cases': 0})
    for model_name, model in plan['models'].items():
        actor = core.Actor().cuda().eval().requires_grad_(False)
        state = torch.load(model['checkpoint'], map_location='cpu', weights_only=False)
        actor.load_state_dict(state['actor'])
        del state
        frozen = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        ctx = mp.get_context('spawn')
        pipes, workers = [], []
        try:
            for index in range(8):
                a, b = ctx.Pipe()
                p = ctx.Process(target=run, args=(b, index, str(OUT / model_name), plan['seed']))
                os.environ['STAGE_PROCESS_CPU'] = str(CPU_POOL[(index + 1) % len(CPU_POOL)])
                p.start()
                os.environ['STAGE_PROCESS_CPU'] = str(CPU_POOL[0])
                b.close()
                pipes.append(a)
                workers.append(p)
            save('workers.json', {'parent': os.getpid(), 'pids': [p.pid for p in workers], 'model': model_name})
            for repetition in range(3):
                observations, tasks = {}, {}
                for i, pipe in enumerate(pipes):
                    pipe.send(('reset', i // 2))
                for i, pipe in enumerate(pipes):
                    row = receive(pipe)
                    task = row['task']
                    assert not task['canonical5m']
                    key = f'phase{i // 2}_side{i % 2}_draw{repetition}'
                    if key in initial_tasks:
                        assert task['signature'] == initial_tasks[key], key
                    else:
                        initial_tasks[key] = task['signature']
                    observations[i], tasks[i] = row['obs'], task
                while observations:
                    assert time.monotonic() < deadline, 'Fixed audit wall limit exceeded'
                    ids = sorted(observations)
                    x = torch.from_numpy(np.asarray([observations[i] for i in ids], dtype=np.float32)).cuda()
                    with torch.no_grad():
                        actions = actor(x).cpu().numpy()
                    for i, action in zip(ids, actions):
                        pipes[i].send(('step', action, 25))
                    for i in ids:
                        row = receive(pipes[i])
                        frames += row['n']
                        assert frames <= plan['maximum_physics_steps']
                        if row['episode'] is None:
                            observations[i] = row['obs']
                        else:
                            episode = row['episode']
                            assert episode['task']['signature'] == tasks[i]['signature']
                            assert not (episode['success'] and episode['collision'])
                            result = {**episode, 'model': model_name, 'phase': i // 2,
                                      'case_id': f'phase{i // 2}_side{i % 2}_draw{repetition}',
                                      'safe_success': episode['success'] and not episode['collision'],
                                      'training': False, 'exploration_noise': False}
                            rows.append(result)
                            with (OUT / 'cases.jsonl').open('a') as f:
                                f.write(json.dumps(result) + '\n')
                            del observations[i]
                    if len(rows) % 8 == 0:
                        save('status.json', {'state': 'RUNNING', 'gradient_updates': 0,
                                             'completed_cases': len(rows), 'physics_steps': frames})
            assert all(torch.equal(frozen[k], v) for k, v in actor.state_dict().items())
            assert sha(model['checkpoint']) == model['sha256']
        finally:
            for pipe in pipes:
                try:
                    pipe.send(('close',))
                except (OSError, EOFError):
                    pass
            for process in workers:
                process.join(5)
                if process.is_alive():
                    process.terminate()
                    process.join(5)
            for pipe in pipes:
                pipe.close()
    assert len(rows) == 48 and len(initial_tasks) == 24
    scores = {}
    for name in plan['models']:
        scores[name] = {}
        for phase in range(4):
            subset = [r for r in rows if r['model'] == name and r['phase'] == phase]
            scores[name][str(phase)] = {'n': len(subset), 'safe_success': sum(r['safe_success'] for r in subset),
                                       'collisions': sum(r['collision'] for r in subset)}
    paired = {'T171_only_success': 0, 'T159_only_success': 0, 'both_success': 0, 'both_fail': 0}
    for key in initial_tasks:
        a = next(r for r in rows if r['case_id'] == key and r['model'] == 'T159')
        b = next(r for r in rows if r['case_id'] == key and r['model'] == 'T171')
        field = 'both_success' if a['safe_success'] and b['safe_success'] else 'T159_only_success' if a['safe_success'] else 'T171_only_success' if b['safe_success'] else 'both_fail'
        paired[field] += 1
    save('summary.json', {'state': 'COMPLETE', 'scores': scores, 'paired_outcomes': paired,
                          'paired_initialization_verified': True, 'actor_unchanged': True,
                          'gradient_updates': 0, 'new_physics_steps': frames,
                          'elapsed_s': time.monotonic() - elapsed_start,
                          'scope': plan['scope'], 'automatic_promotion': False})
    save('status.json', {'state': 'COMPLETE', 'completed_cases': 48, 'new_physics_steps': frames,
                         'training_updates': 0, 'automatic_promotion': False})
    print(json.dumps({'state': 'COMPLETE', 'scores': scores, 'paired_outcomes': paired}), flush=True)


if __name__ == '__main__':
    main()
