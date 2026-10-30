"""Reconstruct T172 initial observations without stepping; preserve censored labels."""
from pathlib import Path
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
import numpy as np

CPU_POOL = [int(x) for x in os.environ.get('STAGE_CPU_POOL', ','.join(map(str, sorted(os.sched_getaffinity(0))))).split(',')]
os.environ['STAGE_CPU_POOL'] = ','.join(map(str, CPU_POOL))
os.sched_setaffinity(0, {int(os.environ.get('STAGE_PROCESS_CPU', str(CPU_POOL[0])))})
import torch

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = REPO / 'stage/runs/t175_paired_advantage'
sys.path.insert(0, str(REPO / 'stage/experiments/t171_reverse_curriculum'))
sys.path.insert(0, str(REPO / 'stage/experiments/t169_value_seed'))
from worker import run
from reward_contract import GAMMA


def main():
    assert not (OUT / 'value_holdout.npz').exists()
    plan = json.loads((HERE / 'holdout_protocol.json').read_text())
    for p, h in plan['input_hashes'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
    rows = [json.loads(x) for x in (REPO / 'stage/runs/t172_frozen_frontier/cases.jsonl').read_text().splitlines()]
    rows = {r['case_id']: r for r in rows if r['model'] == 'T171'}
    core_path = REPO / 'stage/current/outputs/stage_t148/td3_core_goal_contrast.py'
    spec = importlib.util.spec_from_file_location('value_holdout_core', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    actor = core.Actor().cuda().eval().requires_grad_(False)
    ck = torch.load(plan['actor_checkpoint'], map_location='cpu', weights_only=False)
    actor.load_state_dict(ck['actor'])
    del ck
    pipes, processes = [], []
    states, actions, labels, valid, keys = [], [], [], [], []
    try:
        ctx = mp.get_context('spawn')
        for index in range(8):
            a, b = ctx.Pipe()
            p = ctx.Process(target=run, args=(b, index, str(OUT / 'reset_only'), 2026103072))
            os.environ['STAGE_PROCESS_CPU'] = str(CPU_POOL[(index + 1) % len(CPU_POOL)])
            p.start()
            os.environ['STAGE_PROCESS_CPU'] = str(CPU_POOL[0])
            b.close()
            pipes.append(a)
            processes.append(p)
        for repetition in range(3):
            obs = []
            for i, pipe in enumerate(pipes):
                pipe.send(('reset', i // 2))
            for i, pipe in enumerate(pipes):
                assert pipe.poll(60)
                row = pipe.recv()
                assert 'error' not in row, row
                key = f'phase{i // 2}_side{i % 2}_draw{repetition}'
                historic = rows[key]
                assert row['task']['signature'] == historic['task']['signature'], key
                assert abs(row['task']['distance_m'] - historic['task']['distance_m']) < 1e-10
                obs.append(row['obs'])
                keys.append(key)
                terminal = historic['success'] or historic['collision']
                n = historic['physics_steps']
                discount = GAMMA ** n
                value = -.00002 * (1 - discount) / (1 - GAMMA)
                value += GAMMA ** (n - 1) * (10 if historic['success'] else -10 if historic['collision'] else 0)
                value += .3 * historic['task']['distance_m']
                labels.append(value if terminal else np.nan)
                valid.append(terminal)
            x = torch.tensor(np.asarray(obs, dtype=np.float32), device='cuda')
            with torch.no_grad():
                a = actor(x).cpu().numpy()
            states.extend(obs)
            actions.extend(a)
        assert len(states) == 24 and sum(valid) == 16
        np.savez_compressed(OUT / 'value_holdout.npz', states=np.asarray(states, dtype=np.float32),
                            actions=np.asarray(actions, dtype=np.float32),
                            labels=np.asarray(labels), valid=np.asarray(valid), case_ids=np.asarray(keys))
        report = {'state': 'PASS', 'matched_initial_task_hashes': 24, 'complete_mc_labels': 16,
                  'censored_labels_masked': 8, 'new_physics_steps': 0, 'gradient_updates': 0,
                  'note': 'Known same-mapT172 GPUActor rollouts; exact initial reset hashes reconstructed. No unseen-map claim. HistoricalGPU rollout vsT173 CPU rollout numerical policy variants remain a limitation.'}
        (OUT / 'holdout_verification.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    finally:
        for pipe in pipes:
            try:
                pipe.send(('close',))
            except (OSError, EOFError):
                pass
        for p in processes:
            p.join(5)
            if p.is_alive():
                p.terminate()
                p.join(5)
        for pipe in pipes:
            pipe.close()


if __name__ == '__main__':
    main()
