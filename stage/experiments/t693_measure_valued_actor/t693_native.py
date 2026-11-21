"""One intervention, original prefix/time budget, own stochastic continuation."""
from pathlib import Path
import os, sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'stage/experiments/t692_onpolicy_value_causal'))
import t692_native as Prior
C, G, F, E, P, L, np, torch = Prior.C, Prior.G, Prior.F, Prior.E, Prior.P, Prior.L, Prior.np, Prior.torch


class Session(C.Session):
    def __init__(self, job, *args, **kw):
        super().__init__(job, *args, **kw)
        assert L.sha(job['source_path']) == job['source_sha256']
        self.source = E.N.S.arrays(job['source_path'])
        self.branch = int(job['branch_decision']); assert 0 <= self.branch < len(self.source['states'])
        self.branch_command = np.asarray(job['branch_command'], np.float32)
        self.branch_latent = np.asarray(job['branch_latent'], np.float64)
        assert np.array_equal(self.branch_command, self.branch_latent.clip(-1, 1).astype(np.float32))
        assert np.array_equal(self.initial, self.source['initial_native_integration'])
        assert np.array_equal(self.goal, self.source['task_goal_xy'])
        self.decision_integrations = []

    def step(self, forced=None, owner=0, actor_active=True):
        assert forced is None and owner == 0 and actor_active
        i = len(self.rows)
        if i <= self.branch:
            assert np.array_equal(self.packet(), self.source['states'][i]), ('prefix_packet', i)
            assert self.steps == int(self.source['native_counts'][:i].sum())
        self.q.sample = i > self.branch
        assert self.q.rng is not None
        action = self.source['actions'][i] if i < self.branch else (self.branch_command if i == self.branch else None)
        result = F.FieldSession.step(self, action, owner=int(i <= self.branch), actor_active=True)
        self.decision_integrations.append(E.N.K.R.integration(self.env).copy())
        return result


def payload(s, tape=None):
    a = Prior.NORMAL(s, tape)
    a.update(decision_native_integration=np.asarray(s.decision_integrations),
             branch_latent=s.branch_latent.copy())
    return a


def worker(pipe, index, p, counts):
    C.ACTIVE = p; real = G._real_affinity
    def pin(pid, mask):
        return real(pid, {p['stage_cpus'][index]}) if pid == 0 and set(mask) == {index} else real(pid, mask)
    os.sched_setaffinity = pin; G.T.Session = Session
    G.B.native_setup = lambda folder, pc: C.C.consistent_reset.setup(F, folder, pc, C.C.CANONICAL_SETUP)
    P.frozen = L.frozen; E.payload = payload
    G.T.worker(pipe, index, p, counts)


def inspect_prefix(a, source, job):
    k = int(job['branch_decision']); n = len(a['states']); assert 0 <= k < n
    mask = (np.arange(n) <= k).astype(np.uint8)
    assert np.array_equal(a['ownership'], mask) and np.array_equal(a['stochastic'], 1-mask)
    assert np.array_equal(a['initial_native_integration'], source['initial_native_integration'])
    assert np.array_equal(a['task_goal_xy'], source['task_goal_xy'])
    for name in ['states', 'poses', 'history_poses', 'history_goals', 'history_times']:
        assert np.array_equal(a[name][:k+1], source[name][:k+1]), name
    for name in ['next_states', 'next_poses', 'actions', 'native_counts']:
        assert np.array_equal(a[name][:k], source[name][:k]), name
    assert np.array_equal(a['branch_latent'], np.asarray(job['branch_latent'], np.float64))
    assert np.array_equal(a['actions'][k], np.asarray(job['branch_command'], np.float32))
    assert np.array_equal(a['actions'][k], a['branch_latent'].clip(-1, 1).astype(np.float32))
    assert np.array_equal(a['states'][1:], a['next_states'][:-1])
    assert np.array_equal(a['states'][1:, 117:119], a['actions'][:-1])
    assert not a['innovations'][:k+1].any()
    assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
    assert len(a['decision_native_integration']) == n
    assert np.array_equal(a['decision_native_integration'][-1], a['final_native_integration'])
    # Ownership and own proposal differ from original stochastic prefix, so
    # source hold_sha256 must NOT be demanded equal. Physical arrays above and
    # integration arrays in exact regeneration are the applicable contracts.
    return k


def inspect(p, a, row):
    job = row['request']['job']; assert L.sha(job['source_path']) == job['source_sha256']
    source = E.N.S.arrays(job['source_path']); k = inspect_prefix(a, source, job)
    proposals = a['raw_actions'].clip(-1, 1).astype(np.float32)
    assert np.array_equal(a['actions'][k+1:], proposals[k+1:])
    # Generic auditor replays all own carry, raw/clip densities, rewards and
    # history geometry. Forced physical actions were checked separately.
    F.M.audit_full_history(p, dict(row, teacher=False),
                          dict(a, actions=proposals, ownership=np.zeros(len(a['states']), np.uint8)))
    expected = np.random.default_rng(np.random.SeedSequence(row['seed'])).normal(size=a['innovations'][k+1:].shape)
    assert np.array_equal(expected, a['innovations'][k+1:])
    assert int(a['native_counts'].sum()) == row['physics'] <= job['case']['step_cap']
    assert all(L.digest(v) == row['array_hashes'][name] for name, v in a.items())
