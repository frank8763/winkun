"""Matched causal sequence/reset demonstration warm start; zero new RL."""
from pathlib import Path
import sys, copy, io, gzip, json, os, time, shutil, traceback
ROOT = Path(__file__).resolve().parents[3]; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'stage/experiments/t344_neighbour_recovery'))
import t344_core as B
K, T, N, np, torch = B.K, B.T, B.N, B.np, B.torch
read, save, digest, arrays, packed = B.read, B.save, B.digest, B.arrays, B.packed
OUT = ROOT / 'stage/runs/t348_causal_sequence_warmstart'; DOC = ROOT / 'stage/docs/review_20261104'


def verify(p):
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for f, h in p['input_hashes'].items(): assert K.C.sha(f) == h, f
    assert K.S.used(ROOT / 'stage/runs') - K.S.used(OUT) + p['final_bytes'] <= p['global_bytes']
    assert K.S.used(OUT) <= p['final_bytes'] and shutil.disk_usage(ROOT).free >= p['min_free_bytes']


class Actor(torch.nn.Module):
    def __init__(self, p, arm):
        super().__init__(); assert arm in ['reset', 'memory']; self.memory_on = arm == 'memory'
        self.base = K.policy(read(p['base_protocol'])).eval().requires_grad_(False)
        assert digest(self.base.state_dict()) == p['safe_policy_hash'] and not self.base.memory_on
        self.register_buffer('mean', torch.tensor(p['input_mean'], dtype=torch.float64))
        self.register_buffer('scale', torch.tensor(p['input_scale'], dtype=torch.float64))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(p['sequence_seed'])
            self.encoder = torch.nn.Sequential(torch.nn.Linear(120, 32), torch.nn.Tanh()).double()
            self.core = torch.nn.GRU(32, 16).double()
            self.head = torch.nn.Linear(16, 2).double()
            torch.nn.init.zeros_(self.head.weight); torch.nn.init.zeros_(self.head.bias)

    def trainable(self): return [*self.encoder.parameters(), *self.core.parameters(), *self.head.parameters()]

    def learned(self): return {k: v.detach().clone() for k, v in self.state_dict().items() if not k.startswith('base.')}

    def load_learned(self, state): self.load_state_dict(dict(self.state_dict(), **copy.deepcopy(state)), strict=True)

    def phash(self): return digest(dict(memory_on=self.memory_on, state=self.state_dict()))

    def base_actions(self, x):
        with torch.no_grad(): return torch.cat([self.base.step(row[None])[0] for row in x]).double()

    def location(self, b, h): return 1.25 * torch.tanh(torch.atanh(b / 1.25) + self.head(h))

    def step(self, x, hidden=None):
        assert x.shape == (1, 120)
        if hidden is None or not self.memory_on: hidden = x.new_zeros((1, 1, 16), dtype=torch.float64)
        z = self.encoder(((x.double() - self.mean) / self.scale).clamp(-10, 10))
        h, new = self.core(z[None], hidden)
        loc = self.location(self.base_actions(x), h[0])
        return loc.clamp(-1, 1).float(), loc, new, loc

    def sequence(self, x, b):
        z = self.encoder(((x.double() - self.mean) / self.scale).clamp(-10, 10))
        if self.memory_on: h, _ = self.core(z[:, None]); h = h[:, 0]
        else: h, _ = self.core(z[None]); h = h[0]
        return self.location(b, h)


class Learner:
    def __init__(self, p, arm):
        self.actor = Actor(p, arm); self.opt = torch.optim.Adam(self.actor.trainable(), lr=p['sequence_lr']); self.calls = 0
    def state(self): return dict(learned=self.actor.learned(), optimizer=copy.deepcopy(self.opt.state_dict()))
    def load(self, state):
        state = copy.deepcopy(state); self.actor.load_learned(state['learned']); self.opt.load_state_dict(state['optimizer'])
    def hashes(self): return {k: digest(v) for k, v in self.state().items()}


def dataset(p, q, guard):
    groups = []; old = []
    with np.load(p['old_states']) as z: xx = z['states'].copy(); offsets = z['offsets'].copy(); tasks = z['task'].copy()
    assert xx.shape == (28629, 120) and len(tasks) == 211 and np.all(xx[offsets[:-1], 112] == 1)
    base = []
    for start in range(0, len(xx), 512):
        guard(); base.append(q.base_actions(torch.tensor(xx[start:start+512])))
    base = torch.cat(base)
    for i, task in enumerate(tasks):
        sl = slice(int(offsets[i]), int(offsets[i+1])); x = torch.tensor(xx[sl]); b = base[sl]
        old.append(dict(id='old_' + str(i), task=int(task), x=x, b=b, y=b.clone(), start=0))
    for record in p['sequence_sources']:
        a = arrays(record['path']); x = torch.tensor(a['states']); guard()
        assert a['end_flags'][-1, 1] and not a['end_flags'][:, 0].any() and not a['end_flags'][:-1].any()
        groups.append(dict(id=record['name'], x=x, b=q.base_actions(x), y=torch.tensor(a['actions'], dtype=torch.float64), start=record['label_start']))
    return old, groups


def schedule(p, old):
    rng = np.random.default_rng(p['sequence_seed']); result = []
    for k in range(p['fit_steps']):
        ids = [j for j, g in enumerate(old) if g['task'] == k % 8]
        result.append((int(rng.choice(ids)), k % len(p['sequence_sources'])))
    return result


def fit(p, obj, data, guard, event, steps=None):
    old, fresh = data; seq = schedule(p, old); trace = []
    for k, (oi, ni) in enumerate(seq[:p['fit_steps'] if steps is None else steps]):
        guard(); obj.opt.zero_grad(set_to_none=True); losses = []
        for g in [old[oi], fresh[ni]]:
            loc = obj.actor.sequence(g['x'], g['b']); n = g['start']
            loss = .5 * (loc[n:] - g['y'][n:]).square().mean(); assert torch.isfinite(loss)
            loss.backward(); losses.append(float(loss.detach()))
        norm = float(torch.nn.utils.clip_grad_norm_(obj.actor.trainable(), 1., error_if_nonfinite=True))
        event(dict(phase='BEFORE', step=k)); obj.opt.step(); obj.calls += 1
        assert all(v.grad is None for v in obj.actor.base.parameters())
        row = dict(step=k, old_episode=oi, new_source=ni, old_loss=losses[0], new_loss=losses[1], gradient_norm=norm)
        trace.append(row); event(dict(phase='AFTER', step=k))
    return trace


def export(p, obj, path):
    b = io.BytesIO(); torch.save(dict(state=obj.state(), memory_on=obj.actor.memory_on, hashes=obj.hashes(), RL_updates=0), b)
    blob = gzip.compress(b.getvalue(), mtime=0); assert len(blob) <= p['model_bytes']; path.write_bytes(blob)
    return dict(path=str(path), sha256=K.C.sha(path), bytes=len(blob), hashes=obj.hashes(), policy_hash=obj.actor.phash())


def rollout(p, q, job, native, create, witness, deadline, counts, index, previous_total):
    env = None; rows = []; hist = []; hb = []; ha = []; steps = 0; term = trunc = False; info = {}; error = None
    try:
        env, obs, _ = K.R.task_env(create, job); obs = K.C.perturb_anchor(env, obs, job)
        cap = job['case']['step_cap']; assert not env._collision()
        scene = K.C.model_hash(env); goal = env.goal_xy.copy(); initial = K.R.integration(env).copy()
        previous = np.zeros(2, np.float32); pn = 0; hidden = None
        while not (term or trunc) and steps < cap and time.time() < deadline:
            x = K.T.L.packet(env, obs, steps, cap, previous, pn); pose = np.asarray(env._pose()[:3]); meta = witness.get(env)
            assert np.abs(K.H.rebuild(x, pose, meta[0], goal, meta[1]) - x).max() <= 2e-6
            with torch.no_grad():
                before_h = np.zeros(16) if hidden is None else hidden[0, 0].numpy().copy()
                action, _, new, _ = q.step(torch.tensor(x[None]), hidden); action = action[0].numpy()
                hidden = new if q.memory_on else None
                after_h = new[0, 0].numpy().copy()
            before = steps; d0 = float(env._goal_state()[1])
            for _ in range(min(100, cap - steps)):
                if time.time() >= deadline: break
                obs, _, term, trunc, info = native.step_checked(env, action); steps += 1
                if steps % 100 == 0:
                    with counts.get_lock(): counts[index] = previous_total + steps
                if term or trunc: break
            n = steps - before
            if not n: break
            flags = [bool(info.get('collision')), bool(info.get('reach_goal')), bool(trunc)]
            d1 = float(env._goal_state()[1]); z = K.R.GAMMA ** n; terminal = any(flags)
            reward = -.000002 * (1-z)/(1-K.R.GAMMA) + .03*d0 - z*.03*d1*(not terminal)
            if terminal: reward += K.R.GAMMA**(n-1)*(float(flags[1])-float(flags[0])-.5*float(flags[2]))
            nxt = K.T.L.packet(env, obs, steps, cap, action, n)
            rows.append((x, action, n, nxt, pose, np.asarray(env._pose()[:3]), flags, reward)); hist.append(meta); hb.append(before_h); ha.append(after_h)
            previous = action.copy(); pn = n
            assert scene == K.C.model_hash(env) and np.array_equal(goal, env.goal_xy)
        names = ['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards']
        assert rows
        a = {name: np.asarray([r[k] for r in rows]) for k, name in enumerate(names)}; h = hist + [witness.get(env)]
        a['native_counts'] = a['native_counts'].astype(np.uint8); a['end_flags'] = a['end_flags'].astype(np.uint8)
        a.update(history_poses=np.asarray([v[0] for v in h]), history_goals=np.asarray([v[1] for v in h]), history_times=np.asarray([v[2] for v in h]),
                 environment_rewards=a['rewards'].copy(), hidden_before=np.asarray(hb), last_hidden_after=np.asarray(ha[-1]),
                 initial_native_integration=initial, final_native_integration=K.R.integration(env).copy())
        row = dict(id=job['id'], job_id=job['source_reset_id'], role=job['role'], arm=job['arm'], policy_hash=q.phash(), cap=cap,
                   complete=bool(term or trunc), success=bool(info.get('reach_goal')) and not bool(info.get('collision')),
                   collision=bool(info.get('collision')), timeout=bool(trunc), administrative_censor=not bool(term or trunc),
                   physics_steps=steps, remaining_m=float(env._goal_state()[1]), goal_xy=goal.tolist(), scene_sha256=scene,
                   integration_sha256=K.R.arrays_hash(K.R.integration(env)), array_hashes={name:digest(v) for name,v in a.items()})
        return dict(row=row, arrays=a)
    except BaseException:
        if env is not None:
            try:
                partial = dict(final_native_integration=K.R.integration(env).copy(), recorded_physics=np.array([sum(r[2] for r in rows)]), actual_physics=np.array([steps]))
                if rows:
                    for j, name in enumerate(['states','actions','native_counts','next_states','poses','next_poses','end_flags','rewards']):
                        partial[name] = np.asarray([r[j] for r in rows])
                    h = hist + [witness.get(env)]
                    partial.update(hidden_before=np.asarray(hb), last_hidden_after=np.asarray(ha[-1]),
                        history_poses=np.asarray([v[0] for v in h]),history_goals=np.asarray([v[1] for v in h]),history_times=np.asarray([v[2] for v in h]))
                    partial = packed(partial)
                b = io.BytesIO(); np.savez_compressed(b, **partial); blob = b.getvalue()
                assert len(blob) <= p['partial_bytes']
                (OUT / 'workers' / str(index) / 'partial.npz').write_bytes(blob)
            except BaseException:
                save(OUT / 'workers' / str(index) / 'partial_failure.json', dict(error=traceback.format_exc()))
        raise
    finally:
        with counts.get_lock(): counts[index] = previous_total + steps
        if env is not None: witness.events.pop(id(env), None); env.close()


def worker(pipe, index, p, counts, preflight=False):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.optim.Adam.step = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError('No worker optimizer'))
    q = Actor(p, 'reset').eval().requires_grad_(False)
    if preflight:
        cmd = pipe.recv(); q.memory_on = cmd['arm'] == 'memory'; q.load_learned(cmd['learned'])
        hidden = None; output = []
        with torch.no_grad():
            for x in cmd['states']:
                action, loc, new, _ = q.step(torch.tensor(x[None]), hidden); output.append(loc[0].numpy())
                hidden = new if q.memory_on else None
        pipe.send(dict(locations=np.asarray(output), policy_hash=q.phash(), pid=os.getpid())); pipe.close(); return
    folder = OUT / 'workers' / str(index); folder.mkdir(parents=True); log = N.LimitedLog(folder / 'worker.log'); sys.stdout = sys.stderr = log
    total = 0; witness = None
    try:
        m, native, create, witness, reward = B.setup(folder / 'native', p)
        pipe.send(dict(ready=True, pid=os.getpid(), affinity=sorted(os.sched_getaffinity(0))))
        while True:
            job = pipe.recv()
            if job is None: break
            q.memory_on = job['arm'] == 'memory'; q.load_learned(job['learned'])
            if job.get('readback_model'):
                v = torch.load(io.BytesIO(gzip.decompress(Path(job['readback_model']).read_bytes())), weights_only=False, map_location='cpu')
                assert v['memory_on'] == q.memory_on; q.load_learned(v['state']['learned'])
            assert q.phash() == job['policy_hash']
            item = rollout(p, q, job, native, create, witness, job['deadline_epoch'], counts, index, total)
            total += item['row']['physics_steps']; pipe.send(item)
    except EOFError: pass
    except BaseException:
        f = dict(error=traceback.format_exc(), actual_physics=counts[index]); save(folder / 'failure.json', f)
        try: pipe.send(f)
        except OSError: pass
    finally:
        if witness is not None: witness.close()
        pipe.close(); log.close()
