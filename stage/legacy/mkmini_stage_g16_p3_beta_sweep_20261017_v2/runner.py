#!/usr/bin/env python3
"""MK-mini Stage-only G16_P3: research-only four-arm DAAD-SAC night comparison.
A: no modification, B: exploration alpha adjusted by past episode stalls,
C: progress reward shaping, D: both. No planner, no manual recovery actions.
Requires --pilot before --run. No automatic promotion, blind used only once.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, signal, sys, time, traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch

HOME = Path.home()
PR = HOME / 'catkin_ws/src/mk-mini'
ROOT = HOME / 'mkmini_stage_g16_p3_beta_sweep_20261017_v2'
RUN = ROOT / 'run'
D1 = HOME / 'mkmini_stage_g5d1_selector_trainability_v1/run/candidate_offline_research_only.pt'
FROZEN = HOME / 'mkmini_stage_g5f_overnight_20261014_v1/run/D_entropy_retention/resume.pt'
P5 = HOME / 'mkmini_stage_g5g15_p5_stall_aux_v1/result.json'
RUNNER = ROOT / 'runner.py'
TZ = ZoneInfo('Asia/Shanghai')
TRAIN_END = datetime(2026, 10, 17, 14, 15, tzinfo=TZ).timestamp()
HARD_END = datetime(2026, 10, 17, 14, 30, tzinfo=TZ).timestamp()
BRANCHES = (
    'A_beta_000',
    'B_beta_010',
    'C_beta_025',
    'D_beta_050',
)
BETA_BY_BRANCH = {
    'A_beta_000': 0.00,
    'B_beta_010': 0.10,
    'C_beta_025': 0.25,
    'D_beta_050': 0.50,
}
MAX_FRAMES = 3_600_000
CHUNK_FRAMES = 30_000
EVAL_INTERVAL = 240_000
VAL = {'short': (202710811, 202710812, 202710813, 202710814),
       'medium': (202710911, 202710912, 202710913, 202710914)}
BLIND = {'short': (202711811, 202711812, 202711813, 202711814,
                   202711815, 202711816, 202711817, 202711818),
         'medium': (202711911, 202711912, 202711913, 202711914,
                    202711915, 202711916, 202711917, 202711918)}
PROGRESS_BETA = None  # G16-P3 uses BETA_BY_BRANCH
STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    print('G16_P3_STOP_SIGNAL_RECEIVED', flush=True)


signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def save_torch(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def do_prepare():
    path = '/tmp/stage_daad_g5b_selector_recovery_v1'
    sys.path.insert(0, path)
    import g5b
    assert PR.is_dir() and all(x.is_file() for x in (D1, FROZEN, P5))
    p5 = json.loads(P5.read_text())
    assert p5.get('gate') == 'P5_NOT_VALIDATED', 'P5_GATE_UNEXPECTED'
    assert p5.get('policy_updated') is False, 'P5_POLICY_TOUCHED'
    H, C, N, proof, ck, frozen = g5b.prepare(PR)
    assert tuple(C.DURATIONS) == (1, 8, 24)
    assert H.P.TRAIN_HORIZON == {'short': 7000, 'medium': 14000}
    snap = torch.load(FROZEN, map_location='cpu', weights_only=False)
    d1 = torch.load(D1, map_location='cpu', weights_only=False)
    assert snap['format'] == 'G5F_RESUME_V1'
    assert snap['branch'] == 'D_entropy_retention'
    assert int(snap['state']['frames']) == 4_200_000
    assert snap['d1_sha'] == sha(D1)
    assert d1.get('promotion') is False
    protected = {str(x): sha(x) for x in (D1, FROZEN, P5, Path(g5b.__file__))}
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    return g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck


def assert_unchanged(g5b, H, proof, frozen, protected):
    g5b.frozen_assert(proof, frozen, H)
    for name, expected in protected.items():
        assert sha(name) == expected, 'SOURCE_CHANGED:' + name


def agent_new(g5b, H, C, ck, snap, d1, device, seed):
    ag = g5b.new_agent(H, C, ck, device, seed, False)
    for key in ('actor', 'q1', 'q2', 't1', 't2'):
        getattr(ag, key).load_state_dict(snap['net'][key], strict=True)
    # G16_P3 V2 repair: g5b.new_agent returns a frozen research policy.
    # Unfreeze the *new training clone*, never touch the frozen D source.
    ag.actor.train()
    ag.q1.train()
    ag.q2.train()
    for p in ag.actor.parameters():
        p.requires_grad_(True)
    for q in (ag.q1, ag.q2):
        for p in q.parameters():
            p.requires_grad_(True)
    for target in (ag.t1, ag.t2):
        target.eval()
        for p in target.parameters():
            p.requires_grad_(False)
    ag.mean = d1['mean'].to(device)
    ag.std = d1['std'].to(device)
    # All Actor parameters are trainable: not the old duration-only adaptation.
    ag.ap = torch.optim.Adam(ag.actor.parameters(), lr=2e-5)
    ag.cp = torch.optim.Adam(list(ag.q1.parameters()) + list(ag.q2.parameters()), lr=5e-5)
    ag.alpha_d = 0.08
    ag.alpha_c = float(ag.alpha_c)
    ag.updates = 0
    pids = {id(p) for group in ag.ap.param_groups for p in group['params']}
    assert pids == {id(p) for p in ag.actor.parameters()}
    assert all(p.requires_grad for p in ag.actor.parameters())
    return ag


def make_state(name):
    return {'branch': name, 'frames': 0, 'episodes': 0, 'updates': 0,
            'train_goals': {'short': 0, 'medium': 0},
            'train_contacts': {'short': 0, 'medium': 0},
            'duration': {'1': 0, '8': 0, '24': 0},
            'last_eval': 0, 'evaluations': 0, 'best_rank': None,
            'stall_fraction_ema': 0.0, 'base_alpha_c': None,
            'replay_size': 0}


def load_or_new(name, prep):
    g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck = prep
    folder = RUN / name
    folder.mkdir(parents=True, exist_ok=True)
    ag = agent_new(g5b, H, C, ck, snap, d1, device, 202710700)
    replay = C.Replay(12_000)
    state = make_state(name)
    base = float(ag.alpha_c)
    state['base_alpha_c'] = base
    path = folder / 'resume.pt'
    if path.exists():
        x = torch.load(path, map_location='cpu', weights_only=False)
        assert x.get('format') == 'G16_P3_NIGHT_RESUME_V1'
        assert x.get('branch') == name and x['frozen_sha'] == protected[str(FROZEN)]
        assert x['runner_sha'] == sha(RUNNER)
        for k in ('actor', 'q1', 'q2', 't1', 't2'):
            getattr(ag, k).load_state_dict(x['net'][k], strict=True)
        for k in ('ap', 'cp'):
            opt = getattr(ag, k)
            opt.load_state_dict(x['optim'][k])
            for values in opt.state.values():
                for key, val in values.items():
                    if torch.is_tensor(val):
                        values[key] = val.to(device)
        ag.updates = int(x['updates'])
        replay.data.extend(x['replay'])
        state = x['state']
        random.setstate(x['py_rng'])
        np.random.set_state(x['np_rng'])
        torch.set_rng_state(x['torch_rng'])
        if torch.cuda.is_available() and x.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(x['cuda_rng'])
        print('G16_P3_RESUMED', name, state['frames'], flush=True)
    else:
        state['_rng'] = None
    return ag, replay, state


def set_alpha(ag, state, name):
    # G16-P3 isolates progress reward; entropy coefficients remain fixed.
    ag.alpha_c = float(state['base_alpha_c'])
    ag.alpha_d = 0.08


def save_state(name, ag, replay, state, prep):
    protected = prep[8]
    data = {'format': 'G16_P3_NIGHT_RESUME_V1', 'branch': name,
            'frozen_sha': protected[str(FROZEN)], 'runner_sha': sha(RUNNER),
            'net': {k: {n: t.detach().cpu() for n, t in getattr(ag, k).state_dict().items()}
                    for k in ('actor', 'q1', 'q2', 't1', 't2')},
            'optim': {k: getattr(ag, k).state_dict() for k in ('ap', 'cp')},
            'state': state, 'updates': ag.updates, 'replay': list(replay.data),
            'py_rng': random.getstate(), 'np_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    save_torch(RUN / name / 'resume.pt', data)
    write_json(RUN / name / 'progress.json', state)


def run_chunk(name, ag, replay, state, prep, frame_budget, end_time, save=True):
    g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck = prep
    start_frames = state['frames']
    last_save = state['frames']
    while (state['frames'] - start_frames < frame_budget and
           state['frames'] < MAX_FRAMES and time.time() < end_time - 80 and not STOP):
        e = state['episodes']
        task = ('short', 'medium', 'medium', 'medium')[e % 4]
        layout = N.TRAIN_LAYOUTS[(e // 4) % len(N.TRAIN_LAYOUTS)]
        seed = 202710000 + e  # Matched seed and task order across all four branches.
        env = None
        episode_frames = 0
        episode_goal = episode_contact = False
        episode_low_progress = 0
        episode_windows = 0
        mark = 0
        try:
            env, obs, goal = H.P.environment(PR, task, layout, seed)
            goal = np.asarray(goal, dtype=float).reshape(-1)[:2]
            prev = np.asarray(obs, np.float32).copy()
            prev_action = np.zeros(2, np.float32)
            prev_len = 1
            old_pos = np.asarray(env.data.qpos[:2], dtype=float).copy()
            last_mark_dist = float(np.linalg.norm(old_pos - goal))
            last_mark_pos = old_pos.copy()
            set_alpha(ag, state, name)
            while (episode_frames < H.P.TRAIN_HORIZON[task] and
                   state['frames'] < MAX_FRAMES and
                   state['frames'] - start_frames < frame_budget and
                   time.time() < end_time - 80 and not STOP):
                feature = np.asarray(C.feat(obs, prev, prev_action, prev_len), np.float32)
                assert feature.shape == (227,) and np.isfinite(feature).all()
                action, idx, _ = g5b.choice(C, ag, feature, 'natural', stochastic=True, epsilon=.24)
                idx = int(idx)
                assert 0 <= idx < 3
                action = np.asarray(action, np.float32).reshape(2)
                assert np.isfinite(action).all() and np.max(abs(action)) <= 1.00001
                before = np.asarray(obs, np.float32).copy()
                dist0 = float(np.linalg.norm(np.asarray(env.data.qpos[:2], float) - goal))
                ret = 0.; actual = 0; terminated = truncated = False
                n = min(int(C.DURATIONS[idx]), H.P.TRAIN_HORIZON[task] - episode_frames,
                        MAX_FRAMES - state['frames'], frame_budget - (state['frames'] - start_frames))
                for j in range(n):
                    if time.time() >= end_time - 80 or STOP:
                        break
                    nextobs, rew, terminated, truncated, info = N.step_checked(env, action)
                    ret += (ag.gamma ** j) * float(rew)
                    actual += 1
                    episode_frames += 1
                    state['frames'] += 1
                    episode_goal |= bool(info['reach_goal'])
                    episode_contact |= bool(info['collision'])
                    if episode_frames >= mark + 500:
                        pos = np.asarray(env.data.qpos[:2], dtype=float).copy()
                        d = float(np.linalg.norm(pos - goal))
                        if last_mark_dist - d < .03 and np.linalg.norm(pos - last_mark_pos) < .04:
                            episode_low_progress += 1
                        episode_windows += 1
                        last_mark_dist = d
                        last_mark_pos = pos.copy()
                        mark = episode_frames
                    if terminated or truncated:
                        break
                if not actual:
                    break
                dist1 = float(np.linalg.norm(np.asarray(env.data.qpos[:2], float) - goal))
                if BETA_BY_BRANCH[name] > 0.0:
                    # Native SMDP duration discount; progress is observed only after transition.
                    # Collision receives no special positive terminal shaping bonus.
                    shaped = BETA_BY_BRANCH[name] * (dist0 - (ag.gamma ** actual) * dist1)
                    ret += max(-0.08, min(0.08, shaped))
                newf = np.asarray(C.feat(nextobs, before, action, actual), np.float32)
                replay.add(feature, idx, action, ret, newf,
                           ag.gamma ** actual, int(terminated))
                state['duration'][str(C.DURATIONS[idx])] += 1
                # One SAC update per >=16 observed physics steps, identical across arms.
                if len(replay) >= 128:
                    while (ag.updates < state['frames'] // 16 and
                           time.time() < end_time - 90 and not STOP):
                        before_updates = int(ag.updates)
                        losses = ag.update(replay.sample(64))
                        assert all(math.isfinite(float(z)) for z in losses), 'NONFINITE_SAC_LOSS:' + name
                        assert all(torch.isfinite(p).all() for p in ag.actor.parameters())
                        if int(ag.updates) == before_updates:
                            ag.updates += 1
                        assert int(ag.updates) == before_updates + 1, 'UPDATE_COUNTER_INCONSISTENT'
                prev = before
                obs = nextobs
                prev_action = action.copy()
                prev_len = actual
                if terminated or truncated:
                    break
            state['episodes'] += 1
            state['train_goals'][task] += int(episode_goal and not episode_contact)
            state['train_contacts'][task] += int(episode_contact)
            ratio = episode_low_progress / max(1, episode_windows)
            state['stall_fraction_ema'] = .9 * state['stall_fraction_ema'] + .1 * ratio
            state['updates'] = int(ag.updates)
            state['replay_size'] = len(replay)
        finally:
            if env is not None:
                env.close()
        if save and (state['frames'] - last_save >= 30_000 or time.time() >= end_time - 90 or STOP):
            save_state(name, ag, replay, state, prep)
            last_save = state['frames']
        if state['episodes'] % 5 == 0:
            print('G16_P3_TRAIN', json.dumps({'name': name, 'frames': state['frames'],
                 'updates': ag.updates, 'episodes': state['episodes'],
                 'goals': state['train_goals'], 'contacts': state['train_contacts'],
                 'alpha_c': float(ag.alpha_c)}), flush=True)
    if save:
        save_state(name, ag, replay, state, prep)


def evaluate(name, agent, prep, seeds, deadline):
    g5b, H, C, N, *_ = prep
    stats = {}; rows = []
    agent.actor.eval()
    for task, sample_seeds in seeds.items():
        for seed in sample_seeds:
            if time.time() >= deadline - 5 or STOP:
                return None
            limit = time.monotonic() + max(1, deadline - time.time() - 5)
            row = g5b.rollout(H, C, N, agent, PR, task, seed, 'natural', limit)
            if not row.get('complete'):
                return None
            rows.append({'task': task, 'seed': seed, 'safe': int(row['safe_goal']),
                         'contact': int(row['contact']), 'progress_m': float(row['progress_m'])})
        rr = [r for r in rows if r['task'] == task]
        stats[task] = {'n': len(rr), 'safe': sum(r['safe'] for r in rr),
                       'contacts': sum(r['contact'] for r in rr),
                       'mean_progress_m': round(float(np.mean([r['progress_m'] for r in rr])), 5)}
    return {'branch': name, 'stats': stats, 'rows': rows}


def rank_eval(e):
    s = e['stats']
    # Safety first, followed by combined goal-reaching and medium-distance behavior.
    return (-s['short']['contacts'] - s['medium']['contacts'],
            s['short']['safe'] + s['medium']['safe'],
            s['medium']['safe'], s['short']['safe'],
            s['medium']['mean_progress_m'])


def strictly_better(candidate, baseline):
    x, b = candidate['stats'], baseline['stats']
    return ((x['medium']['safe'] > b['medium']['safe'] or
             x['short']['safe'] > b['short']['safe']) and
            x['medium']['safe'] >= b['medium']['safe'] and
            x['short']['safe'] >= b['short']['safe'] and
            x['medium']['contacts'] <= b['medium']['contacts'] and
            x['short']['contacts'] <= b['short']['contacts'])


def best_checkpoint(name, ag, result, state, prep):
    d1 = prep[7]
    z = {'format': 'G16_P3_DEV_RESEARCH_ONLY', 'branch': name, 'frames': state['frames'],
         'frozen_sha': prep[8][str(FROZEN)], 'score': list(rank_eval(result)),
         'mean': d1['mean'].detach().cpu(), 'std': d1['std'].detach().cpu(),
         'actor': {k: v.detach().cpu() for k, v in ag.actor.state_dict().items()},
         'promotion': False, 'blind_used': False}
    save_torch(RUN / name / 'best_dev_research_only.pt', z)
    write_json(RUN / name / 'best_dev.json',
               {'frames': state['frames'], 'branch': name,
                'score': list(rank_eval(result)), 'evaluation': result,
                'promotion': False})


def load_best_agent(prep, name):
    z = torch.load(RUN / name / 'best_dev_research_only.pt', map_location='cpu', weights_only=False)
    assert z['format'] == 'G16_P3_DEV_RESEARCH_ONLY' and z['promotion'] is False
    assert z['frozen_sha'] == prep[8][str(FROZEN)]
    agent = agent_new(prep[0], prep[1], prep[2], prep[10], prep[6], prep[7], prep[9], 202710700)
    agent.actor.load_state_dict(z['actor'], strict=True)
    agent.actor.eval()
    return agent


def source_gate(prep):
    g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck = prep
    assert_unchanged(g5b, H, proof, frozen, protected)
    assert len(N.TRAIN_LAYOUTS) >= 1
    assert set(sum((list(v) for v in VAL.values()), [])) .isdisjoint(
           set(sum((list(v) for v in BLIND.values()), [])))
    assert (RUNNER == Path(__file__).resolve())
    assert sha(RUNNER) == json.loads((ROOT / 'manifest.json').read_text())['runner_sha256']
    print('G16_P3_SOURCE_GATE=PASS ORIGINAL_DAAD_SAC=TRUE PHASE_PLANNER=FALSE', flush=True)


def pilot(prep):
    g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck = prep
    source_gate(prep)
    for name in BRANCHES:
        ag = agent_new(g5b, H, C, ck, snap, d1, device, 202710700)
        # Policy and Q networks must actually update in the FULL SAC algorithm.
        ac_before = {k: v.detach().cpu().clone() for k, v in ag.actor.state_dict().items()}
        q_before = {k: v.detach().cpu().clone() for k, v in ag.q1.state_dict().items()}
        rep = C.Replay(1200)
        state = make_state(name)
        state['base_alpha_c'] = float(ag.alpha_c)
        rr = random.getstate(), np.random.get_state(), torch.get_rng_state()
        random.seed(202710701); np.random.seed(202710701); torch.manual_seed(202710701)
        try:
            # Native MuJoCo *and* SAC optimizer training. No pilot saves into branch history.
            tiny_end = time.time() + 240
            run_chunk(name, ag, rep, state, prep, 3500, tiny_end, save=False)
            print('G16_P3_PILOT_COUNTS', json.dumps({'branch': name,
                  'frames': state['frames'], 'transitions': len(rep),
                  'updates': int(ag.updates), 'min_transitions_for_update': 128,
                  'frames_per_update': 16}), flush=True)
            if ag.updates < 1:
                raise RuntimeError('PILOT_ZERO_UPDATES_CHECK_REPLAY_COUNTS:' + name)
            if not any(not torch.equal(t.detach().cpu(), ac_before[k]) for k, t in ag.actor.state_dict().items()):
                raise RuntimeError('PILOT_ACTOR_NOT_TRAINED:' + name)
            if not any(not torch.equal(t.detach().cpu(), q_before[k]) for k, t in ag.q1.state_dict().items()):
                raise RuntimeError('PILOT_CRITIC_NOT_TRAINED:' + name)
            print('G16_P3_PILOT_ARM=PASS', name, 'frames', state['frames'],
                  'updates', ag.updates, flush=True)
        finally:
            random.setstate(rr[0]); np.random.set_state(rr[1]); torch.set_rng_state(rr[2])
    assert_unchanged(g5b, H, proof, frozen, protected)
    write_json(RUN / 'pilot.json',
               {'gate': 'G16_P3_FOUR_ARM_FOREGROUND_PILOT_PASS',
                'runner_sha256': sha(RUNNER), 'protected': protected,
                'timestamp': datetime.now(TZ).isoformat()})
    print('GATE=G16_P3_FOUR_ARM_FOREGROUND_PILOT_PASS MODEL_PROMOTION=FALSE', flush=True)


def run(prep):
    g5b, H, C, N, proof, frozen, snap, d1, protected, device, ck = prep
    source_gate(prep)
    p = json.loads((RUN / 'pilot.json').read_text())
    assert p['gate'] == 'G16_P3_FOUR_ARM_FOREGROUND_PILOT_PASS'
    assert p['runner_sha256'] == sha(RUNNER)
    assert p['protected'] == protected
    assert time.time() < TRAIN_END - 300, 'NIGHT_TOO_LATE'
    assert not (RUN / 'final.json').exists(), 'ALREADY_FINAL'
    print('G16_P3_NIGHT_WORKER_READY', datetime.now(TZ).isoformat(), flush=True)
    ref = agent_new(g5b, H, C, ck, snap, d1, device, 202710700)
    ref.actor.eval()
    for par in ref.actor.parameters():
        par.requires_grad_(False)
    baseline_path = RUN / 'baseline_development.json'
    if not baseline_path.exists():
        base = evaluate('FROZEN_D', ref, prep, VAL, min(TRAIN_END - 180, time.time() + 500))
        assert base is not None, 'INCOMPLETE_BASELINE'
        write_json(baseline_path, base)
        print('G16_P3_FROZEN_BASELINE', json.dumps(base['stats']), flush=True)
    else:
        base = json.loads(baseline_path.read_text())
        assert base['branch'] == 'FROZEN_D'
    states = {name: load_or_new(name, prep) for name in BRANCHES}
    round_no = 0
    while time.time() < TRAIN_END - 200 and not STOP:
        round_no += 1
        runnable = False
        for name in BRANCHES:
            if STOP or time.time() >= TRAIN_END - 200:
                break
            ag, replay, state = states[name]
            if state['frames'] >= MAX_FRAMES:
                continue
            runnable = True
            random.seed(202710701 + state['episodes'])
            np.random.seed(202710701 + state['episodes'])
            torch.manual_seed(202710701 + state['episodes'])
            before = state['frames']
            run_chunk(name, ag, replay, state, prep, CHUNK_FRAMES, TRAIN_END)
            if state['frames'] > before and (state['frames'] - state['last_eval'] >= EVAL_INTERVAL):
                candidate = evaluate(name, ag, prep, VAL,
                                     min(TRAIN_END - 120, time.time() + 500))
                if candidate is not None:
                    state['evaluations'] += 1
                    state['last_eval'] = state['frames']
                    score = rank_eval(candidate)
                    if state['best_rank'] is None or score > tuple(state['best_rank']):
                        best_checkpoint(name, ag, candidate, state, prep)
                        state['best_rank'] = list(score)
                    save_state(name, ag, replay, state, prep)
                    print('G16_P3_DEV', name, state['frames'],
                          'score', score, 'stats', json.dumps(candidate['stats']), flush=True)
            print('G16_P3_CHUNK_END', name, state['frames'], 'updates', ag.updates,
                  'remaining_seconds', max(0, int(TRAIN_END - time.time())), flush=True)
        if not runnable:
            print('G16_P3_ALL_BRANCHES_HIT_FRAME_BUDGET', flush=True)
            break
    for name, (ag, replay, state) in states.items():
        save_state(name, ag, replay, state, prep)
        if state['frames'] > 0 and state['evaluations'] == 0 and not STOP and time.time() < HARD_END - 1100:
            candidate = evaluate(name, ag, prep, VAL, min(HARD_END - 950, time.time() + 420))
            if candidate is not None:
                best_checkpoint(name, ag, candidate, state, prep)
                state['best_rank'] = list(rank_eval(candidate))
                state['evaluations'] += 1
                state['last_eval'] = state['frames']
                save_state(name, ag, replay, state, prep)
                print('G16_P3_FINAL_DEV', name, json.dumps(candidate['stats']), flush=True)
    # Compare development only; never choose best using blind outcomes.
    rankings = []
    for name in BRANCHES:
        bp = RUN / name / 'best_dev.json'
        if bp.exists():
            rankings.append(json.loads(bp.read_text()))
    rankings.sort(key=lambda z: tuple(z['score']), reverse=True)
    blind_status = 'NOT_USED_NO_DEV_CANDIDATE'
    blind = None
    candidate = None
    if rankings and not STOP and time.time() < HARD_END - 500:
        winner = rankings[0]['branch']
        win_dev = rankings[0]['evaluation']
        if strictly_better(win_dev, base):
            candidate = winner
            chosen = load_best_agent(prep, winner)
            blind_ref = evaluate('FROZEN_D', ref, prep, BLIND, HARD_END - 90)
            blind_cand = evaluate(winner, chosen, prep, BLIND, HARD_END - 90)
            if blind_ref is not None and blind_cand is not None:
                passed = strictly_better(blind_cand, blind_ref)
                blind = {'baseline': blind_ref['stats'], 'candidate': blind_cand['stats'],
                         'strictly_better': passed, 'seeds': BLIND}
                blind_status = 'PASS_RESEARCH_ONLY_NO_PROMOTION' if passed else 'FAIL_NO_PROMOTION'
                print('G16_P3_BLIND', json.dumps(blind), flush=True)
                if passed:
                    # Research candidate only. Never write/replace the historical best.
                    cp = RUN / winner / 'best_dev_research_only.pt'
                    frozen_cp = RUN / 'blind_pass_research_only.pt'
                    if frozen_cp.exists():
                        raise RuntimeError('BLIND_PASS_ALREADY_SAVED')
                    # Re-serialize only the qualified actor, without training checkpoints.
                    blob = torch.load(cp, map_location='cpu', weights_only=False)
                    blob['independent_blind'] = blind
                    blob['promotion'] = False
                    save_torch(frozen_cp, blob)
            else:
                blind_status = 'INCOMPLETE_TIME_LIMIT_NO_PROMOTION'
        else:
            blind_status = 'DEV_NOT_STRICTLY_BETTER_NO_BLIND_USED'
    assert_unchanged(g5b, H, proof, frozen, protected)
    summary = {'gate': 'G16_P3_FOUR_ARM_NIGHT_RESEARCH_COMPLETE',
               'start_date': '2026-10-17', 'hard_end': '2026-10-17T14:30:00+08:00',
               'branch_frames': {n: states[n][2]['frames'] for n in BRANCHES},
               'branch_updates': {n: states[n][0].updates for n in BRANCHES},
               'developed_best': rankings[0]['branch'] if rankings else None,
               'blind_candidate': candidate, 'blind_status': blind_status,
               'blind': blind, 'promotion': False,
               'frozen_sources_unchanged': True}
    write_json(RUN / 'final.json', summary)
    print('G16_P3_NIGHT_FINAL', json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    pa = argparse.ArgumentParser()
    group = pa.add_mutually_exclusive_group(required=True)
    group.add_argument('--pilot', action='store_true')
    group.add_argument('--run', action='store_true')
    args = pa.parse_args()
    if time.time() >= HARD_END:
        raise RuntimeError('PAST_HARD_CUTOFF')
    RUN.mkdir(parents=True, exist_ok=True)
    prep = do_prepare()
    if args.pilot:
        assert not (RUN / 'pilot.json').exists(), 'PILOT_ALREADY_FINISHED'
        pilot(prep)
    else:
        run(prep)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('G16_P3_BLOCKED', repr(exc), flush=True)
        traceback.print_exc()
        sys.exit(2)
