"""Frozen full-continuation versus one-decision substitution; diagnostic only."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'stage/experiments/t651_canonical_outcome_pg'))
import t651_native as Prior

G, F, E, P, L, np, torch = Prior.G, Prior.F, Prior.E, Prior.P, Prior.L, Prior.np, Prior.torch
NORMAL = Prior.C.Prior.Prior.Prior.NORMAL


def intervention(job, decision, source):
    first = job['prefix_first']
    if decision < first:
        return source['actions'][decision], 1
    if decision == first and job['continuation_kind'] == 'candidate_once':
        return np.asarray(job['candidate_action'], np.float32), 1
    return None, 0


class Session(Prior.Session):
    def __init__(self, job, *args, **kwargs):
        super().__init__(job, *args, **kwargs)
        self.job = job
        self.first = job['prefix_first']
        self.source = E.N.S.arrays(job['source_path'])
        assert 0 <= self.first < len(self.source['states'])
        assert not self.source['end_flags'][:self.first].any(), 'Prefix already terminated'
        assert job['remaining_native_steps'] == self.cap-int(self.source['native_counts'][:self.first].sum()) > 0
        assert np.array_equal(self.packet(), self.source['states'][0])
        assert np.array_equal(self.initial, self.source['initial_native_integration'])
        self.forced_actions = []

    def step(self, forced=None, owner=0, actor_active=True):
        assert forced is None and owner == 0 and actor_active
        self.q.sample = False
        k = len(self.rows)
        if k == self.first:
            assert np.array_equal(self.packet(), self.source['states'][k])
            assert self.steps == int(self.source['native_counts'][:k].sum())
            assert self.cap-self.steps == self.job['remaining_native_steps']
        action, ownership = intervention(self.job, k, self.source)
        self.forced_actions.append(np.zeros(2, np.float32) if action is None else np.asarray(action).copy())
        # The loaded actor observes every real packet, including forced-prefix
        # steps. It therefore computes its own carry, also after the one-action
        # intervention. No saved carry is copied between policies.
        return F.FieldSession.step(self, action, owner=ownership, actor_active=True)


def payload(session, tape=None):
    a = NORMAL(session, tape)
    a.update(prefix_first=np.asarray(session.first, np.int64),
             candidate_once=np.asarray(session.job['continuation_kind'] == 'candidate_once', np.uint8),
             forced_actions=np.asarray(session.forced_actions, np.float32))
    return a


def worker(pipe, index, p, counts):
    Prior.ACTIVE = p
    real = G._real_affinity
    def pin(pid, mask):
        return real(pid, {p['stage_cpus'][index]}) if pid == 0 and set(mask) == {index} else real(pid, mask)
    os.sched_setaffinity = pin
    G.T.Session = Session
    G.B.native_setup = lambda folder, pc: Prior.C.consistent_reset.setup(F, folder, pc, Prior.C.CANONICAL_SETUP)
    P.frozen = L.frozen
    assert F.M.P.frozen is L.frozen
    E.payload = payload
    G.T.worker(pipe, index, p, counts)


def inspect(p, a, row):
    job = row['request']['job']
    k = job['prefix_first']
    source = E.N.S.arrays(job['source_path'])
    assert int(a['prefix_first']) == k < len(a['states'])
    assert int(a['candidate_once']) == int(job['continuation_kind'] == 'candidate_once')
    assert not a['end_flags'][:k].any() and not source['end_flags'][:k].any()
    assert job['case']['step_cap']-int(a['native_counts'][:k].sum()) == job['remaining_native_steps']
    assert np.array_equal(a['initial_native_integration'], source['initial_native_integration'])
    for key in ['states', 'poses', 'history_poses', 'history_goals', 'history_times']:
        assert np.array_equal(a[key][:k + 1], source[key][:k + 1]), key
    for key in ['actions', 'native_counts', 'next_states', 'next_poses']:
        assert np.array_equal(a[key][:k], source[key][:k]), key
    expected = np.zeros(len(a['states']), np.uint8)
    expected[:k] = 1
    if job['continuation_kind'] == 'candidate_once':
        expected[k] = 1
        assert np.array_equal(a['actions'][k], np.asarray(job['candidate_action'], np.float32))
    elif job['continuation_kind'] == 'candidate_full':
        assert np.array_equal(a['actions'][k], np.asarray(job['candidate_action'], np.float32))
    assert np.array_equal(a['ownership'], expected)
    assert not a['stochastic'].any()
    proposals = np.clip(a['raw_actions'], -1, 1).astype(np.float32)
    assert np.array_equal(a['actions'][expected == 0], proposals[expected == 0])
    assert np.array_equal(a['states'][1:, 117:119], a['actions'][:-1])
    for n in range(len(a['states'])):
        u, own = intervention(job, n, source)
        assert np.array_equal(a['forced_actions'][n], np.zeros(2, np.float32) if u is None else u)
    assert int(a['native_counts'].sum()) <= job['case']['step_cap']
    # Audit the loaded actor's complete causal recurrence and own proposals;
    # actual forced actions/previous-action observations were checked above.
    F.M.audit_full_history(p, dict(row, teacher=False),
                          dict(a, actions=proposals, ownership=np.zeros_like(expected)))
    assert all(L.digest(v) == row['array_hashes'][key] for key, v in a.items())
