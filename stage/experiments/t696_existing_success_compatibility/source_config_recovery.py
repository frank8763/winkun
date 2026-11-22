"""Fixed seven-source feasibility; no coefficients, optimizer, or native step."""
from pathlib import Path
import copy, gzip, importlib.util, json, os, resource, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'stage/docs/weekend_20261120_23/t696'
DOC = PARENT / 'source_config_recovery'
CAMPAIGN = PARENT.parent
sys.path.insert(0, str(ROOT / 'stage/experiments/t694_full_history_compatibility'))
import t694_actor as A
import t694_wait as Wait


def main():
    started = time.time()
    os.sched_setaffinity(0, {14})
    assert not (DOC / 'PREPARATION.json').exists()
    assert not (DOC / 'PREPARATION_REJECTION.json').exists()
    spec = importlib.util.spec_from_file_location('archive_guard', ROOT / 'stage/experiments/weekend_20261120_23/archive_after_t695.py')
    guard_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard_module)
    guard_module.stopped()
    Wait.Raw.resources()
    inventory = A.read(PARENT / 'SOURCE_INVENTORY.json')
    bound = A.read(DOC / 'BOUND.json')
    (DOC / 'PREPARATION_STARTED.json').write_text(json.dumps(dict(pid=os.getpid(), epoch=time.time())) + '\n')
    p = copy.deepcopy(A.read(ROOT / 'stage/runs/t694_full_history_compatibility/protocol.json'))
    retained = A.read(CAMPAIGN / 'RETAINED_MODEL.json')
    assert A.sha(retained['model']['path']) == retained['model']['sha256']
    old = A.ORIGINAL(p, p['initial']['path'])
    current = A.frozen(p, retained['model']['path'])
    before = A.digest(current.state_dict())
    A.torch.set_num_threads(1)
    A.torch.use_deterministic_algorithms(True)

    def forbidden(*args, **kwargs):
        raise AssertionError('No native/optimizer/Actor fit in preparation')
    A.torch.optim.Adam.step = forbidden
    A.fit = forbidden
    import mujoco
    mujoco.mj_step = forbidden
    last = [0.]

    def guard():
        assert time.time() < min(started + 1150, bound['original_absolute_preparation_stop'] - 10)
        if time.time() - last[0] > 5:
            Wait.Raw.resources()
            last[0] = time.time()

    rows = [dict(id=r['task_id'], path=r['source_path'], sha256=r['sha256'], original384=True)
            for r in p['protected_rows']]
    with gzip.open(ROOT / 'stage/runs/t694_full_history_compatibility/evaluation/cases.jsonl.gz', 'rt') as f:
        by = {r['job_id']: r for r in map(json.loads, f) if r['full'] and not r.get('exact')}
    rows += [dict(id=k, path=by[k]['path'], sha256=by[k]['sha256'], original384=False)
             for k in retained['accepted_gate']['gained']]
    supplement = A.read(CAMPAIGN / 't695/SUPPLEMENTARY_SUCCESS_SET.json')
    rows += [dict(id=r['id'], path=r['path'], sha256=r['sha256'], original384=False)
             for r in supplement['sources']]
    campaign = A.read(CAMPAIGN / 'CAMPAIGN_STATE.json')
    assert len(rows) == 403 and {r['id'] for r in rows} == set(campaign['future_protection_success_ids'])
    protected, old_manifest = [], []
    for index, row in enumerate(rows):
        guard()
        assert A.sha(row['path']) == row['sha256']
        a = A.L.T.R.E.N.S.arrays(row['path'])
        assert not a['ownership'].any() and not a['stochastic'].any()
        assert a['end_flags'][-1, 1] and not a['end_flags'][:-1].any()
        # The retained T694's fixed GRU is the original GRU. Previous audits
        # already certify its commands on all these histories; only features
        # are required to certify a new basis is exactly zero there.
        features, _ = A.trajectory(old, a, row['id'], a['actions'] if row['original384'] else None, guard)
        protected.append(features)
        old_manifest.append(dict(row, decisions=len(features), feature_hash=A.digest(features)))
        if index % 40 == 0:
            print(json.dumps(dict(stage='protected_features', completed=index + 1, elapsed=time.time() - started)), flush=True)
    jobs = {j['id']: j for j in p['evaluation_jobs']}
    neighbor = A.read(ROOT / 'stage/runs/t695_frozen_neighborhood/protocol.json')
    jobs.update({j['id']: j for j in neighbor['neighborhood_jobs']})
    sources, source_manifest = [], []
    physical_fields = ['exact_snapshot', 'delta_x_m', 'delta_y_m', 'delta_yaw_deg',
                       'fullfield_start_pose', 'fullfield_goal_xy', 'circle_shift_x_m']
    old_state = old.state_dict()
    body_keys = [k for k in old_state if not k.startswith(('drive_adapter.', 'steer_adapter.'))]
    for selected in inventory['selected']:
        guard()
        row = selected['row']
        assert selected['id'] not in campaign['future_protection_success_ids']
        assert A.sha(selected['path']) == selected['sha256']
        assert A.read(selected['source_audit'])['state'] == 'PASS'
        actual_job, expected_job = row['request']['job'], jobs[selected['id']]
        assert actual_job['case'] == expected_job['case']
        for key in physical_fields:
            assert actual_job.get(key) == expected_job.get(key), (selected['id'], key)
        a = A.L.T.R.E.N.S.arrays(selected['path'])
        assert all(A.digest(v) == row['array_hashes'][k] for k, v in a.items())
        assert not a['ownership'].any() and not row['teacher']
        assert a['end_flags'][-1, 1] and not a['end_flags'][:-1].any()
        assert int(a['native_counts'].sum()) == row['physics']
        if 'fullfield_start_pose' in expected_job:
            assert bool((abs(a['poses'][:, :2]) <= 7).all())
        source_p = copy.deepcopy(p)
        source_root = Path(selected['source_audit']).parent.parent
        if not (source_root / 'protocol.json').exists():
            source_root = Path(selected['source_audit']).parent
        source_protocol = A.read(source_root / 'protocol.json')
        source_p['raw_sigma'] = source_protocol['raw_sigma']
        source = A.frozen(source_p, row['request']['model'])
        assert all(A.torch.equal(source.state_dict()[k], old_state[k]) for k in body_keys)
        features, mu = A.trajectory(current, a, selected['id'], None, guard)
        target = A.torch.tensor(a['actions'], dtype=A.torch.float64)
        sources.append(dict(id=selected['id'], features=features, base_location=mu, target=target))
        source_manifest.append(dict(id=selected['id'], path=selected['path'], sha256=selected['sha256'],
                                    decisions=len(features), stochastic_executed_labels=bool(a['stochastic'].any()),
                                    feature_hash=A.digest(features), target_hash=A.digest(target)))
        print(json.dumps(dict(stage='source_features', id=selected['id'], decisions=len(features))), flush=True)
    protected = A.torch.cat(protected)
    p.update(maximum_centers=4096, support_fraction=.49, minimum_feature_separation=1e-10)
    print(json.dumps(dict(stage='fixed_geometry', protected_decisions=len(protected),
                         source_decisions=sum(len(s['features']) for s in sources))), flush=True)
    # Existing cardinal construction computes formulas only; it never copies
    # coefficients into an Actor. Correct its old reporting-only fixed count.
    geometry, info = A.construction(p, protected, sources, guard)
    count = sum(len(s['features']) for s in sources)
    info.update(source_decisions=count, exact_duplicate_rows_merged=count - len(geometry['radii']))
    assert A.digest(current.state_dict()) == before
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    assert peak < 4 * 2**30
    result = dict(state='GEOMETRY_SOURCE_AND_403_OWNCARRY_PASS_NO_COEFFICIENT_COMMIT',
                  geometry=info, protected=old_manifest, sources=source_manifest,
                  new_native=0, new_optimizer=0, new_Actor=0, new_Value=0,
                  original_T694_all_parameters_unchanged=True,
                  peak_rss_bytes=peak, elapsed_seconds=time.time() - started,
                  not_complete_entry_preflight=True, not_new_RL_or_generalization=True)
    (DOC / 'PREPARATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['protected', 'sources']}), flush=True)


if __name__ == '__main__':
    started = time.time()
    try:
        main()
    except BaseException:
        (DOC / 'PREPARATION_REJECTION.json').write_text(json.dumps(dict(
            state='PREPARATION_REJECTED_NO_COEFFICIENT_COMMIT', error=traceback.format_exc(),
            new_native=0, new_optimizer=0, new_Actor=0, no_source_seed_radius_or_threshold_rescue=True,
            elapsed_seconds=time.time() - started), indent=2) + '\n')
        raise
    finally:
        p = CAMPAIGN / 'LEDGER.json'
        ledger = A.read(p)
        job = next(j for j in ledger['jobs'] if j['id'] == 'T696_PREPARATION_CONFIG_RECOVERY')
        assert job['state'] == 'PREPARATION_RESERVED_NO_RUN_OR_COEFFICIENT_COMMIT'
        elapsed = time.time() - started
        job.update(state='PREPARATION_PASS' if (DOC / 'PREPARATION.json').exists() else 'PREPARATION_REJECTED',
                   science_seconds=elapsed, original_upper_seconds=1200)
        ledger['science_seconds'] += elapsed
        p.write_text(json.dumps(ledger, indent=2) + '\n')
