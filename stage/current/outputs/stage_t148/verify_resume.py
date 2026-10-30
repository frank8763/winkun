"""Verify every saved component before allowing any optimizer update."""
from pathlib import Path
import importlib.util
import inspect
import json
import os
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
os.environ['T148_RESTORE_ONLY'] = '1'
spec = importlib.util.spec_from_file_location('t148_verify', ROOT / 'train.py')
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b

original_emit = t.emit
verified = {}

def emit(name, obj):
    if name in ('T148_PARENT_FULL_RESUME=PASS', 'TD3_T6_RESUME=PASS'):
        scope = inspect.currentframe().f_back.f_locals
        saved = scope.get('parent_state', scope.get('state'))
        agent = scope['agent']
        for key in ('actor', 'actor_target', 'critic', 'critic_target',
                    'actor_optimizer', 'critic_optimizer'):
            verified[key] = same(saved[key], getattr(agent, key).state_dict())
        for key in ('frames', 'decisions', 'episode_index', 'episodes', 'probes'):
            verified[key] = same(saved[key], scope[key])
        verified['updates'] = agent.total_updates == saved['updates']
        verified['replay'] = same(saved['replay'], list(scope['replay']))
        verified['numpy_rng'] = same(saved['rng_state'], scope['rng'].bit_generator.state)
        verified['torch_rng'] = same(saved['torch_rng_state'], torch.get_rng_state())
        assert all(verified.values()), verified
        (ROOT / 'restore_exact_verification.json').write_text(json.dumps({
            'result': 'PASS', 'checks': verified, 'training_started': False,
            'frames': scope['frames'], 'replay_size': len(scope['replay']),
        }, indent=2) + '\n')
        print('T148_EXACT_STATE_VERIFIED=PASS', json.dumps(verified), flush=True)
    original_emit(name, obj)

t.emit = emit
t.main()
assert verified, 'No restoration was verified'
