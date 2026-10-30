"""Read-only coverage audit of the exact T148 replay mixture."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'early_paired/plan.json').read_text())
OUTPUT = ROOT / 'replay_audit.json'
HISTORY = Path('/home/linghu/mkmini_stage_td3_t41_dual_replay_foundation_20261027_v1/t16_frozen_experience.pt')
SUCCESS = Path('/home/linghu/mkmini_stage_td3_t42_verified_3m_replay_20261027_v1/verified_3m_success.npz')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def coverage(rows):
    s = np.asarray([r[0] for r in rows], dtype=np.float32)
    r = np.asarray([r[2] for r in rows], dtype=np.float64)
    d = np.asarray([r[4] for r in rows], dtype=np.float64)
    dist, sn, cs = s[:, 48] * 10, s[:, 49], s[:, 50]
    clear = s[:, :48].min(1) * 6
    return {
        'n': len(rows), 'initial_task_side': dict(Counter(x[5] for x in rows)),
        'goal_left_fraction': float(np.mean(sn < -0.12)),
        'goal_right_fraction': float(np.mean(sn > 0.12)),
        'near_goal_020_045_fraction': float(np.mean((dist >= .20) & (dist <= .45))),
        't118_turn_mask_fraction': float(np.mean((dist >= .20) & (dist <= 1.2) &
            (cs >= 0) & (np.abs(sn) >= .12) & (np.abs(sn) <= .8) & (clear >= .42))),
        't118_near_mask_fraction': float(np.mean((dist >= .20) & (dist <= .45) &
            (cs < .9) & (np.abs(sn) >= .25) & (clear >= .42))),
        'reward_quantiles': dict(zip(['min','p10','median','p90','max'], map(float,np.quantile(r,[0,.1,.5,.9,1])))),
        'terminal_transitions': int(d.sum()),
    }

def main():
    assert sha(HISTORY) == '440e3887596e6900dde36e59c69d1823be6cf6948b3c92ac3e4c60e9eafb6de0'
    assert sha(SUCCESS) == '743c3b3099031a0f4104e2e41af22c12d491a70a6f3d780ca34477463cf9192b'
    obj = torch.load(HISTORY, map_location='cpu', weights_only=False)
    hist = obj['replay']
    good = []
    offset = 0
    for ep in obj['episodes']:
        n = int(ep['physics_frames'])
        if ep['success'] and not ep['collision']:
            good.extend(hist[offset:offset+n])
        offset += n
    assert offset == len(hist)
    with np.load(SUCCESS, allow_pickle=False) as z:
        s,a,r,ns,d = [z[k].copy() for k in ['states','raw_actions','rewards','next_states','dones']]
        offsets = z['episode_offsets'].tolist()
    assert offsets == [0,10612,20319]
    success = [(s[i],a[i],float(r[i]),ns[i],float(d[i]),'LEFT' if i < 10612 else 'RIGHT') for i in range(len(s))]
    banks = {'old_random': coverage(hist), 'old_success': coverage(good),
             'success_left': coverage(success[:10612]), 'success_right': coverage(success[10612:])}
    models = {}
    weights = {'online':.5,'old_success':.125,'old_random':.125,'success_left':.125,'success_right':.125}
    for name, spec in PLAN['models'].items():
        assert sha(spec['checkpoint']) == spec['sha256']
        ck = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        online = coverage(ck['replay'])
        all_banks = {'online': online, **banks}
        expected = {key: sum(weights[k] * all_banks[k][key] for k in weights) for key in [
            'goal_left_fraction','goal_right_fraction','near_goal_020_045_fraction',
            't118_turn_mask_fraction','t118_near_mask_fraction']}
        expected['initial_task_LEFT_fraction'] = sum(weights[k] * all_banks[k]['initial_task_side'].get('LEFT',0) / all_banks[k]['n'] for k in weights)
        models[name] = {'checkpoint_sha256': spec['sha256'], 'frames':ck['frames'],
            'online_bank': online, 'expected_full_batch_fractions': expected,
            'optimizer_lrs': {'actor':[x['lr'] for x in ck['actor_optimizer']['param_groups']],
                              'critic':[x['lr'] for x in ck['critic_optimizer']['param_groups']]}}
    dt, gamma = .002, .99
    report = {'experiment':'STAGE_T149_READ_ONLY_REPLAY_AND_TIME_AUDIT','training':False,
        'batch_weights': weights, 'frozen_banks': banks, 'models':models,
        'time_scale':{'measured_step_s':dt,'gamma':gamma,'efold_steps':-1/math.log(gamma),
            'efold_seconds':-dt/math.log(gamma),'half_life_seconds':dt*math.log(.5)/math.log(gamma),
            'terminal_reward_discount_at_7500_steps':gamma**7500,
            'note':'Terminal-reward attenuation alone is not proof of failure; dense rewards and actor auxiliary objectives also matter.'},
        'limitations':['Labels denote initial task side, not current goal-bearing sign.',
            'Old and new reward distributions need not be stationary; quantiles alone do not establish comparability.',
            'Coverage is not a causal explanation for navigation failure.'],
        'sources':{'time_discretization':'https://proceedings.mlr.press/v97/tallec19a.html',
                   'experience_replay':'https://proceedings.mlr.press/v119/fedus20a.html'}}
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'models':models,'time_scale':report['time_scale']},indent=2))

if __name__ == '__main__':
    main()
