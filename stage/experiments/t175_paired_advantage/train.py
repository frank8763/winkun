"""Two critic controls: nominalMC targets with/without paired action-return supervision."""
from pathlib import Path
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import time
import numpy as np

os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = REPO / 'stage/runs/t175_paired_advantage'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(path, value):
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(partial, path)


def used():
    return sum(p.stat().st_size for p in (REPO / 'stage/runs').rglob('*') if p.is_file())


def main():
    lock = (OUT / 'process.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (OUT / 'status.json').exists(), 'No duplicate or implicit restart'
    plan = json.loads((HERE / 'protocol.json').read_text())
    for p, digest in plan['input_hashes'].items():
        assert sha(p) == digest, p
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    spec = importlib.util.spec_from_file_location('alignment_core', REPO / 'stage/current/outputs/stage_t148/td3_core_goal_contrast.py')
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    parent = torch.load(plan['parent_checkpoint'], map_location='cpu', weights_only=False)
    replay = torch.load(plan['replay_checkpoint'], map_location='cpu', weights_only=False)['replay']
    collision = np.asarray([i for i, r in enumerate(replay) if r[4] and r[2] < 0])
    goals = np.asarray([i for i, r in enumerate(replay) if r[4] and r[2] > 0])
    with np.load(plan['demonstrations'], allow_pickle=False) as z:
        names = ['states', 'actions', 'rewards', 'next_states', 'dones', 'discounts']
        demo_arrays = [z[n].astype(np.float32).copy() for n in names]
    demos = [tuple(a[i] for a in demo_arrays) for i in range(len(demo_arrays[0]))]
    traces = {}
    for name in plan['mc_train_courses'] + plan['mc_development_return_holdout_courses']:
        with np.load(REPO / 'stage/runs/t173_q_audit' / ('mc_' + name + '.npz'), allow_pickle=False) as z:
            traces[name] = {k: z[k].copy() for k in ['states', 'actions', 'observed_returns']}
    native_rows = [json.loads(x) for x in (REPO / 'stage/runs/t173_q_audit/results.jsonl').read_text().splitlines()]
    branches = [r for r in native_rows if r['kind'] == 'branch']
    pair_train = [r for r in branches if r['prefix_steps'] in (0,10000) and r['intervention'] != 'baseline']
    pair_states, pair_actions, pair_bases, pair_deltas = [], [], [], []
    for row in pair_train:
        base = next(r for r in branches if r['case']['id'] == row['case']['id'] and r['prefix_steps'] == row['prefix_steps'] and r['intervention'] == 'baseline')
        assert base['branch']['state_sha256'] == row['branch']['state_sha256']
        pair_states.append(row['branch']['observation']); pair_actions.append(row['branch']['action']); pair_bases.append(base['branch']['action'])
        pair_deltas.append(row['return_from_branch'] - base['return_from_branch'])
    ps, pa, pb = [torch.tensor(np.asarray(v,dtype=np.float32), device='cuda') for v in (pair_states,pair_actions,pair_bases)]
    py = torch.tensor(np.asarray(pair_deltas,dtype=np.float32)[:,None],device='cuda')
    assert len(pair_train)==18
    with np.load(plan['initial_state_return_holdout'],allow_pickle=False) as z:
        mask=z['valid']; vs=torch.tensor(z['states'][mask],device='cuda'); va=torch.tensor(z['actions'][mask],device='cuda'); vy=z['labels'][mask].copy()
    assert len(vy)==16
    actor = core.Actor().cuda().eval().requires_grad_(False)
    actor.load_state_dict(parent['actor'])
    frozen_actor = {k: v.clone() for k, v in actor.state_dict().items()}
    start = time.monotonic()
    deadline = start + plan['wall_seconds']
    all_results = {}

    def evaluate(critic):
        result = {'terminal_errors': {}, 'mc_courses': {}, 'branch_predictions': []}
        for kind, ids in [('collision', collision), ('goal', goals)]:
            s = torch.tensor(np.asarray([replay[i][0] for i in ids]), device='cuda')
            a = torch.tensor(np.asarray([replay[i][1] for i in ids]), device='cuda')
            labels = np.array([replay[i][2] for i in ids])
            with torch.no_grad():
                q1, q2 = critic(s, a)
            q1, q2 = q1.cpu().numpy().ravel(), q2.cpu().numpy().ravel()
            result['terminal_errors'][kind] = {'n': len(ids), 'q1_mae': float(np.abs(q1 - labels).mean()),
                                              'minq_mae': float(np.abs(np.minimum(q1, q2) - labels).mean())}
        for name, trace in traces.items():
            predictions = []
            with torch.no_grad():
                for i in range(0, len(trace['states']), 256):
                    s = torch.tensor(trace['states'][i:i + 256], device='cuda')
                    a = torch.tensor(trace['actions'][i:i + 256], device='cuda')
                    p1, p2 = critic(s, a)
                    predictions.append(torch.cat([p1, p2], dim=1).cpu().numpy())
            q = np.concatenate(predictions)
            labels = trace['observed_returns']
            result['mc_courses'][name] = {'role': 'return_label_holdout' if name in plan['mc_development_return_holdout_courses'] else 'mc_train',
                                         'q1_mae': float(np.abs(q[:, 0] - labels).mean()),
                                         'minq_mae': float(np.abs(q.min(axis=1) - labels).mean()),
                                         'q1_bias': float((q[:, 0] - labels).mean()),
                                         'initial_q1': float(q[0, 0]), 'initial_mc': float(labels[0])}
        for row in branches:
            with torch.no_grad():
                p1, p2 = critic(torch.tensor([row['branch']['observation']], device='cuda'),
                                torch.tensor([row['branch']['action']], device='cuda'))
            result['branch_predictions'].append({'job_id': row['job_id'], 'case': row['case']['id'],
                                                  'prefix_steps': row['prefix_steps'],
                                                  'intervention': row['intervention'],
                                                  'q1': float(p1), 'q2': float(p2),
                                                  'complete_mc_return': row['return_from_branch']})
        comparisons = []
        for row in result['branch_predictions']:
            if row['intervention'] == 'baseline':
                continue
            baseline = next(r for r in result['branch_predictions'] if r['case'] == row['case'] and
                            r['prefix_steps'] == row['prefix_steps'] and r['intervention'] == 'baseline')
            predicted = row['q1'] - baseline['q1']
            observed = row['complete_mc_return'] - baseline['complete_mc_return']
            comparisons.append({'case': row['case'], 'prefix_steps': row['prefix_steps'],
                                'intervention': row['intervention'], 'q1_gain': predicted,
                                'mc_gain': observed, 'informative': abs(predicted) > 1e-6 and abs(observed) > 1e-5,
                                'sign_agreement': predicted * observed > 0})
        result['action_rank_comparisons'] = comparisons
        with torch.no_grad():
            v1,v2=critic(vs,va)
        result['initial_state_return_holdout_q1_mae']=float(np.abs(v1.cpu().numpy().ravel()-vy).mean())
        result['mean_return_holdout_q1_mae']=result['initial_state_return_holdout_q1_mae']
        held=[r for r in comparisons if r['prefix_steps']==5000 and r['informative']]
        result['paired_action_return_holdout']={'correct':sum(r['sign_agreement'] for r in held),'n':len(held)}
        plus5 = [r for r in comparisons if r['case'].startswith('5m_') and r['intervention'] == 'q_plus' and r['informative']]
        result['fixed_5m_qplus_candidate_sign_agreement'] = {'correct': sum(r['sign_agreement'] for r in plus5), 'n': len(plus5)}
        result['critic_gate'] = (result['terminal_errors']['collision']['q1_mae'] <= 1 and
                                result['terminal_errors']['goal']['q1_mae'] <= 1 and
                                result['mean_return_holdout_q1_mae'] <= 2 and
                                result['paired_action_return_holdout']['correct'] >= 7)
        return result

    for arm_name, arm in plan['arms'].items():
        assert used() + 128 * 1024**2 < 1024**3
        folder = OUT / arm_name
        folder.mkdir()
        torch.manual_seed(plan['seed'])
        torch.cuda.manual_seed_all(plan['seed'])
        uniform_rng, terminal_rng, demo_rng, mc_rng = [np.random.default_rng(plan['seed'] + i) for i in range(4)]
        critic = core.Critic().cuda()
        critic.load_state_dict(parent['critic'])
        target = core.Critic().cuda().eval().requires_grad_(False)
        target.load_state_dict(parent['critic_target'])
        actor_target = core.Actor().cuda().eval().requires_grad_(False)
        actor_target.load_state_dict(parent['actor_target'])
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
        optimizer.load_state_dict(copy.deepcopy(parent['critic_optimizer']))
        for group in optimizer.param_groups:
            group['lr'] = 1e-4
        before = evaluate(critic)
        for update in range(1, plan['critic_updates_per_arm'] + 1):
            assert time.monotonic() < deadline
            ids = uniform_rng.choice(len(replay), 192, replace=False)
            bad = terminal_rng.choice(collision, 48, replace=True)
            good = terminal_rng.choice(goals, 48, replace=True)
            if arm['terminal_balanced']:
                ids = np.r_[ids[:96], bad, good]
            batch = [replay[int(i)] for i in ids]
            batch += [demos[int(i)] for i in demo_rng.choice(len(demos), 64, replace=False)]
            s, a, r, ns, d, g = [torch.tensor(np.asarray([row[k] for row in batch], dtype=np.float32), device='cuda') for k in range(6)]
            with torch.no_grad():
                na = (actor_target(ns) + (torch.randn_like(a) * .15).clamp(-.4, .4)).clamp(-1, 1)
                t1, t2 = target(ns, na)
                y = r[:, None] + (1 - d[:, None]) * g[:, None] * torch.minimum(t1, t2)
            q1, q2 = critic(s, a)
            td_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            # Every arm draws and computes the same MC auxiliary batches; only its coefficient differs.
            mc_s, mc_a, mc_y = [], [], []
            for name in plan['mc_train_courses']:
                trace = traces[name]
                indices = np.r_[0, min(1,len(trace['states'])-1), mc_rng.integers(0,len(trace['states']),14)]
                mc_s.extend(trace['states'][indices])
                mc_a.extend(trace['actions'][indices])
                mc_y.extend(trace['observed_returns'][indices])
            ms = torch.tensor(np.asarray(mc_s), device='cuda')
            ma = torch.tensor(np.asarray(mc_a), device='cuda')
            my = torch.tensor(np.asarray(mc_y, dtype=np.float32)[:, None], device='cuda')
            m1, m2 = critic(ms, ma)
            mc_loss = F.mse_loss(m1, my) + F.mse_loss(m2, my)
            p1,p2=critic(ps,pa); b1,b2=critic(ps,pb)
            pair_loss=F.mse_loss((p1-b1)/.05,py/.05)+F.mse_loss((p2-b2)/.05,py/.05)
            loss = td_loss + arm['mc_weight'] * mc_loss + arm['pair_weight'] * pair_loss
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 10)
            optimizer.step()
            if update % 2 == 0:
                with torch.no_grad():
                    for tnet, net in [(target, critic), (actor_target, actor)]:
                        for tp, p in zip(tnet.parameters(), net.parameters()):
                            tp.mul_(.995).add_(p, alpha=.005)
            if update % 100 == 0:
                row = {'update': update, 'td_loss': float(td_loss.detach()), 'mc_loss': float(mc_loss.detach()), 'pair_loss':float(pair_loss.detach())}
                with (folder / 'losses.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                save(OUT / 'status.json', {'state': 'RUNNING', 'arm': arm_name, 'critic_updates_this_arm': update,
                                          'actor_updates': 0, 'new_physics_steps': 0, 'elapsed_s': time.monotonic() - start})
        after = evaluate(critic)
        assert all(torch.equal(frozen_actor[k], v) for k, v in actor.state_dict().items())
        checkpoint = folder / 'critic_alignment.pt'
        partial = checkpoint.with_suffix('.partial')
        torch.save({'actor': actor.state_dict(), 'actor_target': actor_target.state_dict(),
                    'critic': critic.state_dict(), 'critic_target': target.state_dict(),
                    'critic_optimizer': optimizer.state_dict(), 'critic_updates': plan['critic_updates_per_arm'],
                    'actor_updates': 0, 'new_physics_steps': 0, 'parent_checkpoint_sha256': sha(plan['parent_checkpoint']),
                    'protocol_sha256': sha(HERE / 'protocol.json'), 'arm': arm_name,
                    'replay': 'External immutable T171 parent; no online transitions added',
                    'control_hold_steps': 25, 'discount_efold_seconds': 60.,
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    'numpy_rngs': [g.bit_generator.state for g in (uniform_rng, terminal_rng, demo_rng, mc_rng)]}, partial)
        assert partial.stat().st_size < 64 * 1024**2
        os.replace(partial, checkpoint)
        all_results[arm_name] = {'before': before, 'after': after, 'checkpoint': str(checkpoint.relative_to(REPO)),
                                 'checkpoint_sha256': sha(checkpoint), 'actor_unchanged': True,
                                 'actor_updates': 0, 'new_physics_steps': 0}
        save(folder / 'result.json', all_results[arm_name])
        print(json.dumps({'arm': arm_name, 'terminal': after['terminal_errors'],
                          'return_holdout_q1_mae': after['mean_return_holdout_q1_mae'], 'pair_holdout':after['paired_action_return_holdout'],
                          'gate': after['critic_gate']}), flush=True)
        del critic, target, actor_target, optimizer
    save(OUT / 'summary.json', {'state': 'COMPLETE', 'arms': all_results, 'automatic_promotion': False,
                               'actor_updates': 0, 'new_physics_steps': 0,
                               'note': 'Critic-only repair. IdenticalActor implies unchanged navigation behavior; no new5m success claim.'})
    save(OUT / 'status.json', {'state': 'COMPLETE', 'arms_completed': len(all_results),
                              'critic_updates_total': len(all_results) * plan['critic_updates_per_arm'],
                              'actor_updates': 0, 'new_physics_steps': 0,
                              'elapsed_s': time.monotonic() - start, 'automatic_promotion': False})


if __name__ == '__main__':
    main()
