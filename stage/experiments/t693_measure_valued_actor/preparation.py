"""Bound real sources and verify algebra; NOT the complete launch preflight.

This writes only preparation documents, never a run, checkpoint or protocol.
Canonical Session/worker, native audit and end-to-end recovery remain required.
"""
from pathlib import Path
import copy, hashlib, json, os, resource, sys, time, traceback
import t693_estimator as E
ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT/'stage/docs/weekend_20261120_23/t693'
sys.path.insert(0, str(ROOT/'stage/experiments/t690_score_weighted_baseline'))
import t690_learning as L
sys.path.insert(0, str(ROOT/'stage/experiments/t692_onpolicy_value_causal'))
import t692_wait as Wait
np, torch, M, T = L.np, L.torch, L.M, L.T


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def main():
    start = time.time(); os.sched_setaffinity(0, {14})
    assert not (ROOT/'stage/runs/t693_measure_valued_actor').exists()
    assert not (DOC/'PREPARATION.json').exists(), 'Do not repeat consumed preparation'
    assert (DOC.parent/'T693_METHOD_SELECTION_PRIMARY_REVIEW.md').exists()
    Wait.Raw.resources()
    previous = ROOT/'stage/runs/t692_onpolicy_value_causal'
    report, sealed = L.read(previous/'FINAL_REPORT.json'), L.read(previous/'SEALED.json')
    pids = sealed['scientific_pids']+[report['helper_pid'], L.read(previous/'DRIVER_STARTED.json')['pid']]
    assert sealed['all_scientific_processes_stopped'] and all(not Path('/proc', str(v)).exists() for v in pids)
    p = L.read(ROOT/'stage/runs/t690_score_weighted_baseline/protocol.json')
    # Slots and every random draw are fixed before loading source arrays or Y.
    fixedslots = E.slots(); random_draws = [[E.draws(i, j) for j in range(4)] for i in range(8)]
    contract = E.algebra_contract()
    def forbidden(*a, **kw):
        raise AssertionError('Preparation forbids optimizer/native calls')
    torch.optim.Adam.step = forbidden
    if 'mujoco' in sys.modules: sys.modules['mujoco'].mj_step = forbidden
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    q, opt = L.make(p); before = L.optimizer_state(q, opt); initial = L.digest(before)
    groups = L.B.dataset(p); jobs = {j['id']: j for j in p['evaluation_jobs']}
    records = []; probe = None
    for i, key in enumerate(E.IDS):
        for rep, row in enumerate(groups[key][:4]):
            assert L.sha(row['path']) == row['sha256'] and not row.get('exact')
            a = T.R.E.N.S.arrays(row['path'])
            assert all(L.digest(v) == row['array_hashes'][k] for k, v in a.items())
            assert row['policy_hash'] == q.phash() and not a['ownership'].any() and a['stochastic'].all()
            assert np.array_equal(a['actions'], a['raw_actions'].clip(-1, 1).astype(np.float32))
            expected = np.random.default_rng(np.random.SeedSequence(row['seed'])).normal(size=a['innovations'].shape)
            assert np.array_equal(expected, a['innovations'])
            n = len(a['states']); assert 0 < n <= E.SLOTS
            slot = int(fixedslots[i, rep]); active = slot < n
            g = M.features(q, a, key, row['sha256'])
            with torch.no_grad(): mu = M.location(q, g).numpy()
            assert float(np.abs(mu-a['locations']).max()) < 1e-10
            entry = dict(task=key, rep=rep, slot=slot, length=n, active=active,
                         source=row['path'], source_sha256=row['sha256'], seed=row['seed'],
                         step_cap=jobs[key]['case']['step_cap'],
                         prefix_physics=int(a['native_counts'][:slot].sum()) if active else None,
                         mu=mu[slot].tolist() if active else None)
            if active:
                entry['commands'] = {arm: [[E.command(mu[slot], z).tolist() for z in pair]
                                           for pair in random_draws[i][rep][arm]] for arm in ['mvd', 'score']}
                if probe is None: probe = (g, slot)
            records.append(entry)
    assert len(records) == 32 and not opt.state
    numeric = None
    if probe is not None:
        g, slot = probe; variables = [v for v in q.parameters() if v.requires_grad]
        assert sum(v.numel() for v in variables) == 11906
        theta = torch.cat([v.detach().flatten() for v in variables]).clone()
        def objective():
            # Synthetic detached credit tests the actual raw-mean chain rule;
            # it is not a new physical return or measured Actor gradient.
            return (M.location(q, g)[slot]*torch.tensor([.7, -.3], dtype=torch.float64)).sum()
        gradient = torch.cat([v.flatten() for v in torch.autograd.grad(objective(), variables)]).detach()
        direction = torch.tensor(np.random.default_rng([E.SEED, 4]).normal(size=11906))
        direction /= direction.norm()
        def assign(vec):
            k = 0
            with torch.no_grad():
                for v in variables:
                    v.copy_(vec[k:k+v.numel()].reshape_as(v)); k += v.numel()
        try:
            assign(theta+1e-6*direction); plus = float(objective().detach())
            assign(theta-1e-6*direction); minus = float(objective().detach())
        finally: assign(theta)
        analytic, finite = float(gradient@direction), (plus-minus)/2e-6
        error = abs(analytic-finite); assert error < 1e-8+1e-4*abs(analytic)
        numeric = dict(analytic=analytic, numeric=finite, error=error, synthetic_credit_only=True)
    assert L.digest(L.optimizer_state(q, opt)) == initial and not opt.state
    old = [r for r in records if r['active'] and not r['task'].startswith('t502_')]
    field = [r for r in records if r['active'] and r['task'].startswith('t502_')]
    collection = 16*sum(r['step_cap'] for r in records if r['active'])
    full = 2*p['child_max_physics']; rate = p['admission_physics_per_second']
    bound = dict(collection_physics=collection, collection_commands=16*sum(r['active'] for r in records)+6,
                 candidate_physics=full, candidate_commands=2*p['child_max_commands'],
                 total_physics=collection+full, admission_physics_per_second=rate,
                 physical_admission_seconds=(collection+full)/rate,
                 original_5700_main_fits_physics_admission=(collection+full)/rate < 5700)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    result = dict(state='PREPARATION_CONTRACT_PASS_NOT_LAUNCH_PREFLIGHT',
                  algebra=contract, records=records, numeric=numeric,
                  source32_arrays_hashes_owncarry_and_innovation_exact=True,
                  support=dict(active_old=len(old), active_field=len(field), passed=bool(old and field)),
                  proposed_bounds=bound, actual_Actor=0, actual_Value=0, actual_optimizer=0, actual_native=0,
                  all_Actor_Adam_restored=True, peak_rss_bytes=peak, elapsed_seconds=time.time()-start,
                  pending=['actual Session/worker ownership and audit corruption contracts',
                           'runtime and sequential children', 'complete zero-update entrance preflight',
                           'independent reservation', 'exact GitHub source', 'unique launch'])
    save(DOC/'PREPARATION.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['algebra', 'records']}))


if __name__ == '__main__':
    started = time.time()
    try: main()
    except BaseException:
        save(DOC/'PREPARATION_ERROR.json', dict(error=traceback.format_exc(), elapsed_seconds=time.time()-started,
                                              native=0, optimizer=0, do_not_repeat_without_review=True))
        raise
