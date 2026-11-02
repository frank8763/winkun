"""Fixed-policy, native interventional three-action value pilot.

The value means a prescribed open-loop three-action sequence followed by the
same frozen T299 deterministic feedback policy, with the original six-metre
goal, clock and termination. This is neither a CRL reproduction nor on-policy
AR data. No Actor is optimized or selected here.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t304_native_chunk_value'
DOC = ROOT / 'stage/docs/distance_20261102'
sys.path.insert(0, str(ROOT / 'stage/experiments/t302_balanced_bridge_learning'))
import t302_learning as T
from t280_worker import generate, LimitedLog
from t291_codec import pack_full, unpack_full
sys.path.insert(0, str(ROOT / 'stage/experiments/t278_fixed_goal_starts'))
from t278_guard import Budget
import copy, contextlib, gzip, hashlib, io, json, os, shutil, time, traceback
np, torch, C, S, R = T.np, T.torch, T.C, T.S, T.R


def read(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path) as z:
        return unpack_full({k: z[k].copy() for k in z.files})


def actor(p):
    q = T.policy(read(p['source_protocol']))
    q.load_state_dict(T.L.read(p['source_checkpoint'])['policy'])
    assert S.digest(q.state_dict()) == p['policy_hash']
    return q.eval().requires_grad_(False)


def verify(p):
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for path, digest in p['input_hashes'].items():
        assert C.sha(path) == digest, path
    assert S.used(OUT) < p['final_bytes'] and S.used(ROOT / 'stage/runs') < p['global_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']


class ForcedChunk:
    def __init__(self, q, job):
        self.q, self.job, self.decision = q, job, 0

    def step(self, x):
        result = list(self.q.step(x))
        j = self.decision - self.job['branch_decision']
        if 0 <= j < 3:
            result[0] = torch.tensor([self.job['forced_chunk'][j]], dtype=result[0].dtype)
        self.decision += 1
        return tuple(result)


def worker(pipe, index, p, counts):
    folder = OUT / 'workers' / str(index)
    folder.mkdir(parents=True)
    log = LimitedLog(folder / 'worker.log')
    sys.stdout = sys.stderr = log
    total = 0
    try:
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        native, create = S.native_setup(folder)
        q = actor(p)
        pipe.send(dict(ready=True, pid=os.getpid(), policy_hash=S.digest(q.state_dict()),
                       CPU_affinity=sorted(os.sched_getaffinity(0))))
        while True:
            job = pipe.recv()
            if job is None:
                break
            proxy = ForcedChunk(q, job)
            item = generate(job, proxy, native, create, job['deadline_epoch'], counts, index, total)
            total += item['row']['physics_steps']
            item['row'].update(policy_hash=p['policy_hash'], family=job['family'], arm=job['arm'],
                branch_decision=job['branch_decision'], forced_chunk=job['forced_chunk'],
                not_on_policy_data=True, fixed_deterministic_continuation=True)
            pipe.send(item)
    except EOFError:
        pass
    except BaseException:
        failure = dict(error=traceback.format_exc(), completed_worker_physics=total)
        C.save(folder / 'failure.json', failure)
        try:
            pipe.send(failure)
        except OSError:
            pass
    finally:
        C.save(folder / 'log_accounting.json', dict(bytes_observed=log.total, bytes_written=log.written))
        pipe.close(); log.close()


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(126, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))

    def forward(self, x, blind=False):
        if blind:
            x = torch.cat([x[:, :120], torch.zeros_like(x[:, 120:])], dim=1)
        return self.net(x)[:, 0]


def value_data(p, rows):
    x, labels, families, arms, details = [], [], [], [], []
    for source in p['sources']:
        rr = [next(r for r in rows if r['family'] == source['id'] and r['arm'] == a) for a in p['arms']]
        aa = [arrays(r['path']) for r in rr]
        b = source['branch_decision']
        initial = aa[1]['states'][b]
        assert all(np.array_equal(a['states'][b], initial) for a in aa)
        values = [float(S.monte_carlo(a['environment_rewards'] - .1 * a['aux_costs'], a['native_counts'])[b]) for a in aa]
        for arm, row, a, value in zip(p['arms'], rr, aa, values):
            x.append(np.r_[initial, np.asarray(source['chunks'][arm], np.float32).reshape(-1)])
            labels.append(value - values[1]); families.append(source['id']); arms.append(arm)
            details.append(dict(family=source['id'], arm=arm, value_at_branch=value,
                center_value=values[1], advantage=value-values[1], success=row['success'],
                collision=row['collision'], timeout=row['timeout'], remaining_m=row['remaining_m']))
    x = np.asarray(x, np.float32); labels = np.asarray(labels, np.float64)
    train = np.array([not f.endswith('_24') for f in families])
    assert train.sum() == 18 and (~train).sum() == 6
    mean = x[train].astype(np.float64).mean(0)
    scale = np.maximum(x[train].astype(np.float64).std(0), 1e-3)
    target_scale = max(float(np.sqrt(np.mean(labels[train]**2))), 1e-6)
    features = np.clip((x-mean)/scale, -10, 10).astype(np.float32)
    return dict(x=features, raw_x=x, y=(labels/target_scale).astype(np.float32), raw_y=labels,
        train=train, families=families, arms=arms, details=details,
        normalization=dict(mean=mean, scale=scale, target_scale=target_scale))


def fit(data, p, guard, event, holder=None):
    torch.manual_seed(p['seed'])
    initial = Head()
    models = {k: copy.deepcopy(initial) for k in ['action', 'blind']}
    optimizers = {k: torch.optim.Adam(q.parameters(), lr=p['head_learning_rate']) for k,q in models.items()}
    if holder is not None:
        holder.update(models=models, optimizers=optimizers, initial=S.digest(initial.state_dict()))
    x, y = torch.tensor(data['x']), torch.tensor(data['y'])
    train = torch.tensor(data['train'])
    losses = []
    for step in range(p['head_steps']):
        guard(); record = []
        for name, q in models.items():
            opt = optimizers[name]; opt.zero_grad()
            loss = (q(x[train], name == 'blind') - y[train]).square().mean()
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(q.parameters(), 5., error_if_nonfinite=True)
            event(name, step); opt.step(); record.append(float(loss.detach()))
        losses.append(record)
    assert torch.equal(models['blind'].net[0].weight[:, 120:], initial.net[0].weight[:, 120:])
    with torch.no_grad():
        pred = {n: q(x, n=='blind').numpy().astype(np.float64)*data['normalization']['target_scale'] for n,q in models.items()}
    return models, optimizers, np.asarray(losses, np.float64), pred, S.digest(initial.state_dict())


def metrics(data, pred, p):
    result = {}
    for name, values in pred.items():
        test = ~data['train']; y = data['raw_y']
        families = []
        for family in sorted({f for f,t in zip(data['families'], test) if t}):
            ix = np.flatnonzero(np.asarray(data['families']) == family)
            truth, guess = y[ix], values[ix]
            pairs = [(i,j) for i,j in [(0,1),(0,2),(1,2)] if abs(truth[i]-truth[j]) > p['value_margin']]
            correct = sum((truth[i]-truth[j])*(guess[i]-guess[j]) > 0 for i,j in pairs)
            best = int(np.argmax(guess))
            families.append(dict(id=family, MSE=float(np.mean((guess-truth)**2)),
                informative_pairs=len(pairs), correct_pairs=int(correct),
                top_choice_correct=bool(truth.max()-truth[best] <= p['value_margin']),
                true_advantages=truth.tolist(), predicted_advantages=guess.tolist(),
                native_outcomes=[data['details'][i] for i in ix]))
        result[name] = dict(test_MSE=float(np.mean((values[test]-y[test])**2)), families=families)
    a,b = result['action'],result['blind']
    checks = dict(test_MSE_at_most_half_blind=a['test_MSE'] <= .5*b['test_MSE'],
        each_heldout_family_MSE_not_over102pct=all(x['MSE'] <= 1.02*y['MSE'] for x,y in zip(a['families'],b['families'])),
        both_heldout_families_informative=all(x['informative_pairs'] >= 2 for x in a['families']),
        all_informative_pair_signs_correct=all(x['correct_pairs'] == x['informative_pairs'] for x in a['families']),
        both_top_choices_correct=all(x['top_choice_correct'] for x in a['families']))
    result['screen'] = dict(checks=checks, passed=all(checks.values()),
        automatic_Actor_update=False, limitation='Two already-consumed reset-family holdouts only; no global Q calibration or navigation stability claim.')
    return result
