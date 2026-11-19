"""Prepare immutable continuations and real-sequence zero-update contracts."""
from pathlib import Path
import ast
import copy
import gzip
import json
import math
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / 'stage/docs/day_20261119/t659'
sys.path.insert(0, str(ROOT / 'stage/experiments/t645_iterative_recovery_relay'))
import control as Resource


def main():
    resources = Resource.resources()
    import t659_runtime as T
    assert not T.OUT.exists()
    DOC.mkdir(exist_ok=True, parents=True)
    p = T.read(ROOT / 'stage/runs/t656_qualified_recovery_pg/protocol.json')
    initial_inputs = copy.deepcopy(p['inputs'])
    candidate = T.read(ROOT / 'stage/runs/t656_qualified_recovery_pg/pg/summary.json')['proposal']
    assert T.read(ROOT / 'stage/runs/t656_qualified_recovery_pg/pg/audit.json')['state'] == 'PASS'
    reference = p['initial']
    jobs = {j['id']: j for j in p['evaluation_jobs']}
    offline = T.read(ROOT / 'stage/runs/t646_offline_night/OFFLINE_INPUTS.json')
    # Choose without reading either frozen candidate's outcome: lexicographic
    # four original successful IDs absent from the previous regression screen.
    unmeasured_ids = sorted({r['id'] for r in offline['members'] if r['id'].startswith('dev_task6_')}
                            - set(p['old_screen_ids']))[:4]
    assert len(unmeasured_ids) == 4
    unmeasured = set(unmeasured_ids)
    baseline_ids = ['dev_task6_00', 'dev_task6_02', 'dev_task6_03', 'dev_task6_26',
                    'dev_task3_00', 't502_field_sector5_offset1', *unmeasured_ids,
                    'wide117_80_dev6_LEFT_18', 'wide117_8m_dev6_RIGHT_18']
    sources = {r['id']: dict(path=r['offline_path'], sha256=r['sha256']) for r in offline['members']}
    sources.update({r['job_id']: r for r in p['field_keep_rows'] if r['job_id'] not in sources})
    old_review = T.read(ROOT / 'stage/docs/day_20261119/t656/ACTUAL_T656_REVIEW.json')
    p.update(id='T659_FROZEN_CONTINUATION_REPLACEMENT_GATE', out=str(T.OUT), arm='panel', arms=['panel'],
             authorization_approved=False, reference=reference, candidate=candidate, initial=reference,
             all_seconds=3600, main_seconds=1800, audit_seconds=300, backup_seconds=600, source_seconds=240,
             max_bytes=128*2**20, global_bytes=10*2**30, child_bytes=128*2**20,
             protocol_revision=2, preparation_only=True,
             new_Actor=0, new_Critic=0, new_optimizer=0, preflight_optimizer=0,
             new_Actor_Adam=0, shadow_Actor_Adam=0, independent_historical_Actor_Adam=0,
             new_RL=0, new_teacher=0, Actor_each_cap=0, learning_seconds=0,
             user_authorization='USER_EXPLICITLY_DEFERRED_LAUNCH_REVISE_ONLY',
             independent_predictor_calibration=False, no_conditional_learning=True)
    # Match the worker's adapter-aware loader before calling its leaf auditor.
    T.Native.P.frozen = T.Native.L.frozen
    assert T.Native.F.M.P.frozen is T.Native.L.frozen
    for key in ['registration_epoch', '_root_sha256', 'absolute_all_stop', 'absolute_native_stop', 'absolute_learning_stop']:
        p.pop(key, None)
    q = T.R.L.frozen(p, candidate['path'])
    assert q.phash() == candidate['policy_hash']
    queries = []
    cached_sources = {}
    max_carry_error = 0.
    loss_error, grad_error = 0., 0.

    def query(key, source, first, category):
        nonlocal max_carry_error, loss_error, grad_error
        assert T.sha(source['path']) == source['sha256']
        a = T.R.E.N.S.arrays(source['path'])
        assert 0 <= first < len(a['states'])
        x = T.torch.tensor(a['states'][:first+1])
        q.reset_noise()
        q.sample = False
        with T.torch.no_grad():
            location = q.sequence(x)[-1]
            predicted = location.clamp(-1, 1).float().numpy()
            hidden = None
            for packet in x:
                action, loc, hidden, unused = q.step(packet[None], hidden)
            error = float((location - loc[0]).abs().max())
            max_carry_error = max(max_carry_error, error)
            assert error < 1e-12 and T.np.array_equal(predicted, action[0].numpy())
        # Exact-algebra control with a real causal observation as features.
        # This scalar fixture is not a trained/validated success predictor.
        w = T.torch.tensor([.1, -.2, .3, -.4], dtype=T.torch.float64, requires_grad=True)
        f = (w * x[-1, :4].double()).sum()
        y, b = f.new_tensor(float(first == 0)), f.new_tensor(.375)
        direct = (f-y).square()
        residual = ((f-b)-(y-b)).square()
        g1 = T.torch.autograd.grad(direct, w, retain_graph=True)[0]
        g2 = T.torch.autograd.grad(residual, w)[0]
        loss_error = max(loss_error, float((direct-residual).abs().detach()))
        grad_error = max(grad_error, float((g1-g2).abs().max()))
        assert loss_error < 1e-12 and grad_error < 1e-12
        job = copy.deepcopy(jobs[key])
        job.update(query_key=f'{key}__prefix{first}', prefix_first=first, source_path=source['path'],
                   source_sha256=source['sha256'], source_category=category,
                   candidate_action=predicted.tolist(),
                   remaining_native_steps=int(job['case']['step_cap']-a['native_counts'][:first].sum()),
                   initial_contract=dict(scene_sha256=bytes(a['scene_sha256']).hex(),
                       integration_sha256=T.Native.E.N.K.R.arrays_hash(a['initial_native_integration']),
                       initial120_sha256=T.Native.E.N.K.R.arrays_hash(a['states'][0]),
                       goal_xy=a['task_goal_xy'].tolist()))
        assert job['remaining_native_steps'] > 0
        queries.append(job)
        cached_sources[source['path']] = source['sha256']

    for key in baseline_ids:
        query(key, sources[key], 0, 'unmeasured_frozen_pair' if key in unmeasured else 'known_selected_diagnostic')
    for row in old_review['comparisons']:
        query(row['id'], dict(path=row['origin_path'], sha256=row['origin_sha256']),
              row['first_actual_departure']['decision'], 'known_selected_actual_departure')
    assert len(queries) == 16 and len({j['query_key'] for j in queries}) == 16
    assert q.phash() == candidate['policy_hash']
    assert not T.torch.cuda.is_initialized()
    q.reset_noise()
    p['queries'] = queries
    case_files = [ROOT/'stage/runs/t653_afternoon_outcome_constraints/mean/cases.jsonl.gz',
                  ROOT/'stage/runs/t656_qualified_recovery_pg/pg/cases.jsonl.gz']
    indexes = [[json.loads(v) for v in gzip.decompress(f.read_bytes()).decode().splitlines()] for f in case_files]
    cached_measurements = []
    for query_job in queries:
        if query_job['source_category'] == 'unmeasured_frozen_pair':
            continue
        kinds = ['reference'] if query_job['prefix_first'] else ['reference','candidate_full']
        for kind in kinds:
            pool = indexes[0 if kind == 'reference' else 1]
            matching = [r for r in pool if r['full'] and r['job_id'] == query_job['id']]
            # Fixed deterministic existing exact20, no selection by success.
            row = next(r for r in matching if r['role']=='exact20')
            if query_job['prefix_first']:
                assert row['path'] == query_job['source_path']
            assert T.sha(row['path']) == row['sha256']
            cached_sources[row['path']] = row['sha256']
            cached_measurements.append(dict(query_key=query_job['query_key'],kind=kind,source_row=row))
    p['cached_measurements'] = cached_measurements
    assert len(cached_measurements) == 20 and len({v['source_row']['path'] for v in cached_measurements}) == 16
    covered = {(r['query_key'],r['kind']) for r in cached_measurements}
    new_specs = [(j,kind) for j in queries for kind in ['reference','candidate_once','candidate_full']
                 if (j['query_key'],kind) not in covered]
    p['new_measurement_count'] = len(new_specs)
    assert p['new_measurement_count'] == 28
    p['max_physics'] = 2*sum(j['case']['step_cap'] for j,kind in new_specs)
    p['max_commands'] = 2*p['new_measurement_count']+6
    p['total_max_physics'], p['total_max_commands'], p['total_main_seconds'] = p['max_physics'], p['max_commands'], p['main_seconds']
    storage = p['new_measurement_count']*(p['fresh_bytes']+p['compact_bytes']) + p['full_bytes'] + 16*2**20
    assert storage < p['max_bytes']
    total_used = T.R.used(ROOT / 'stage/runs')
    assert total_used+p['max_bytes'] < p['global_bytes']
    prior_paths = [ROOT/'stage/runs'/name for name in ['t654_verified_deviation_recovery',
                    't655_joint_current_recovery','t656_qualified_recovery_pg']]
    ledgers = [T.read(path/'LEDGER.json') for path in prior_paths]
    p['prior_physics'] = sum(v['physics'] for v in ledgers)
    p['prior_commands'] = sum(v['commands'] for v in ledgers)
    p['prior_main_seconds'] = sum(v['main_seconds'] for v in ledgers)
    p['prior_storage_bytes'] = sum(T.R.used(path) for path in prior_paths)
    assert p['prior_physics']+p['max_physics'] <= p['combined_physics_cap']
    assert p['prior_commands']+p['max_commands'] <= p['combined_command_cap']
    assert p['prior_main_seconds']+p['main_seconds'] <= p['combined_main_cap']
    assert p['prior_storage_bytes']+p['max_bytes'] <= p['combined_storage_cap']
    # Test ownership at the prefix and exact one-decision switch, no simulator.
    fixture = dict(actions=T.np.asarray([[.1,.2],[.3,.4]], T.np.float32))
    for kind in ['reference','candidate_once','candidate_full']:
        job = dict(prefix_first=2, continuation_kind=kind, candidate_action=[.5,.6])
        assert [T.Native.intervention(job, k, fixture)[1] for k in range(4)] == [1,1,int(kind=='candidate_once'),0]
    # Exercise the actual Session.step dispatch without creating an env. The
    # base physics step is replaced only within this preflight try/finally.
    from types import SimpleNamespace
    saved_step = T.Native.F.FieldSession.step
    def fake_step(session, action, owner=0, actor_active=True):
        session.calls.append((action is not None, owner, actor_active, session.q.sample))
        session.rows.append(None)
        session.steps += 100
    try:
        T.Native.F.FieldSession.step = fake_step
        for kind in ['reference','candidate_once','candidate_full']:
            session = object.__new__(T.Native.Session)
            session.job = dict(prefix_first=2, continuation_kind=kind, candidate_action=[.5,.6],remaining_native_steps=800)
            session.first = 2
            session.source = dict(actions=fixture['actions'], states=T.np.zeros((3,120),T.np.float32),
                                  native_counts=T.np.asarray([100,100,100]))
            session.rows, session.calls, session.forced_actions, session.steps = [], [], [], 0
            session.cap = 1000
            session.q = SimpleNamespace(sample=True)
            session.packet = lambda: T.np.zeros(120,T.np.float32)
            for unused in range(4):
                session.step()
            assert [v[1] for v in session.calls] == [1,1,int(kind=='candidate_once'),0]
            assert all(v[0] == bool(v[1]) and v[2] and not v[3] for v in session.calls)
    finally:
        T.Native.F.FieldSession.step = saved_step
    # Real full-history auditor fixtures: a forced action equal to the source
    # actor's own action cannot change the recorded physics. This tests all
    # ownership/carry/audit branches; it is not a new counterfactual outcome.
    actual_audit_fixtures = 0
    for evidence in old_review['comparisons']:
        source = T.R.E.N.S.arrays(evidence['origin_path'])
        k = evidence['first_actual_departure']['decision']
        for kind in ['reference','candidate_once','candidate_full']:
            job = dict(jobs[evidence['id']], prefix_first=k, source_path=evidence['origin_path'],
                       continuation_kind=kind, candidate_action=source['actions'][k].tolist(),
                       remaining_native_steps=int(jobs[evidence['id']]['case']['step_cap']-source['native_counts'][:k].sum()))
            ownership = T.np.zeros(len(source['states']),T.np.uint8)
            ownership[:k] = 1
            if kind == 'candidate_once':
                ownership[k] = 1
            forces = T.np.zeros_like(source['actions'])
            forces[:k] = source['actions'][:k]
            if kind == 'candidate_once':
                forces[k] = source['actions'][k]
            a = dict(source, ownership=ownership, prefix_first=T.np.asarray(k,T.np.int64),
                     candidate_once=T.np.asarray(kind=='candidate_once',T.np.uint8),forced_actions=forces)
            row = dict(request=dict(job=job,model=reference['path']), seed=None, teacher=False,
                       array_hashes={key:T.digest(value) for key,value in a.items()})
            T.Native.inspect(p,a,row)
            actual_audit_fixtures += 1
    # Exercise the actual reuse validation: source record identity, own-carry
    # recomputation, full arrays and exact action at the candidate fork.
    reused = T.cached(p,lambda:None)
    assert len(reused) == 20 and all(r['historical_reuse'] for r in reused)
    for file in HERE.glob('*.py'):
        ast.parse(file.read_text())
    files = set(HERE.glob('*.py')) | {DOC/'PLAN.md', DOC/'PRIMARY_REVIEW.md', DOC/'REVISION.md',DOC/'USER_DECISION.json',
            DOC/'EVIDENCE_PREFLIGHT.json',
            ROOT/'stage/runs/t656_qualified_recovery_pg/protocol.json',
            ROOT/'stage/runs/t656_qualified_recovery_pg/pg/summary.json',
            ROOT/'stage/runs/t656_qualified_recovery_pg/pg/audit.json',
            ROOT/'stage/docs/day_20261119/t656/ACTUAL_T656_REVIEW.json'}
    files.update(Path(j['exact_snapshot']) for j in queries)
    files.update(Path(m['path']) for m in [reference, candidate, p['safe_retained']])
    files.update(path/'LEDGER.json' for path in prior_paths)
    files.update(case_files)
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name and str(name).startswith(str(ROOT)) and str(name).endswith('.py'):
            files.add(Path(name).resolve())
    p['inputs'] = initial_inputs | cached_sources | {str(f): T.sha(f) for f in files}
    for f, h in p['inputs'].items():
        assert T.sha(f) == h, f
    # Native/controller defaults are inherited; unrelated old data panels and
    # learning allocations do not belong in this frozen diagnostic protocol.
    for key in ['reuse_rows','evaluation_jobs','memory_sources','field_keep_rows','training_rows',
                'reference_by_id','screen_jobs','reconstruction_references','field_reconstruct_references',
                'profile_jobs','field_jobs','training_jobs','key_jobs','existing_field_full_rows',
                'field_reconstruct_jobs','recovery_jobs','original_jobs','field_train_jobs',
                'target_reference','initial_corrections','initial_reference','initial_field_keep_rows']:
        p.pop(key,None)
    T.save(HERE/'protocol.json', p)
    T.save(DOC/'RESOURCE_PREFLIGHT.json', resources)
    T.save(DOC/'PREFLIGHT.json', dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER', new_native=0, new_optimizer=0,
        protocol_revision=2,preparation_only=True,
        frozen_two_actual_models_checked=True, own_carry_real_sequence_error=max_carry_error,
        subtraction_MSE_loss_error=loss_error, subtraction_MSE_gradient_error=grad_error,
        subtraction_alone_not_a_method_gain=True, full_causal_candidate_first_actions_verified=True,
        actual_Session_step_dispatch_checked=True, actual_full_history_auditor_fixtures=actual_audit_fixtures,
        query_count=16, fresh_measurements=p['new_measurement_count'], exact_replays=p['new_measurement_count'],
        historical_reused_entries=20,historical_unique_files=16,historical_bytes_and_own_carry_reaudited=True,
        scientific_predictor_not_trained=True,
        commands_worst=p['max_commands'], physics_worst=p['max_physics'], storage_worst=storage,
        global_worst=total_used+p['max_bytes'], native_object_not_created=True))
    T.save(DOC/'BUDGET.json', {k:p[k] for k in ['all_seconds','main_seconds','audit_seconds','backup_seconds','source_seconds',
         'max_physics','max_commands','max_bytes','global_bytes','new_Actor','new_Critic','new_optimizer',
         'prior_physics','prior_commands','prior_main_seconds','prior_storage_bytes','combined_physics_cap',
         'combined_command_cap','combined_main_cap','combined_storage_cap']})
    print(json.dumps(dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER', queries=16,
                         max_physics=p['max_physics'], max_commands=p['max_commands'], storage_worst=storage)))


if __name__ == '__main__':
    main()
