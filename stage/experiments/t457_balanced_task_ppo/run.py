"""One task-balanced on-policy batch; existing head-only Actor and auxiliary data.

This changes coverage, batch size and empirical normalization jointly. It is
not a matched-compute proof or a new algorithm, and never resumes an old run.
"""
from pathlib import Path
import copy, importlib.util, io, json, math, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t457_balanced_task_ppo'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage457_pinned455', ROOT/'stage/experiments/t455_frozen_representation_ppo/run.py')
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
D, P, F, M, R, E, torch = H.D, H.P, H.F, H.M, H.R, H.E, H.torch
np = D.np
read, save, sha, digest = P.read, P.save, P.sha, P.digest


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def worker(pipe, index, p, counts):
    M.worker(pipe, index, p, counts)


class Pool(M.Pool):
    def __init__(self, p, counts, guard):
        self.guard, self.children, self.pipes = guard, [], []
        ctx = R.M.mp.get_context('spawn')
        try:
            for i in range(8):
                a, b = ctx.Pipe()
                c = ctx.Process(target=worker, args=(b, i, p, counts))
                c.start()
                b.close()
                self.children.append(c)
                self.pipes.append(a)
            self.ready = [self.receive(i) for i in range(8)]
            assert all(v['state'] == 'READY' for v in self.ready)
            f = Path(p['out'])/'worker_generations.json'
            records = read(f) if f.exists() else []
            records.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, records)
        except BaseException:
            self.close()
            raise


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def objects(p, parent=False):
    obj = H.make(p)
    model = p['anchor_parent'] if parent else p['sampling_initial']
    P.load(obj, P.checkpoint(model['path'])['joint_state'])
    assert digest(P.state(*obj)) == model['joint_hash']
    assert sum(v.numel() for v in obj[0].parameters() if v.requires_grad) == 130
    return obj


def auxiliary(p):
    parent = objects(p, parent=True)
    groups, own = D.anchors(p, parent[0])
    corrections = [D.correction(r) for r in p['correction_rows']]
    return groups, own, corrections


def fit(obj, arrays, p, groups, own, corrections, guard, event):
    q, v, ao, vo = obj
    frozen = H.frozen_digest(obj)
    device = next(q.parameters()).device
    assert len(arrays) == p['batch_episodes'] == 22
    batch, errors = [], []
    for i, a in enumerate(arrays):
        assert not a['ownership'].any() and a['stochastic'].all()
        assert np.array_equal(a['states'][1:], a['next_states'][:-1])
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        x = torch.tensor(a['states'], device=device)
        raw = torch.tensor(a['raw_actions'], device=device)
        atom = torch.tensor(a['atom'], device=device)
        old = torch.tensor(a['logp'], device=device)
        with torch.no_grad():
            loc = q.sequence(x)
            er = float((loc-loc.new_tensor(a['locations'])).abs().max())
            lp = float((P.L.density(loc, raw, atom, p['raw_sigma'])-old).abs().max())
            assert er < 1e-10 and lp < 1e-9, (i, er, lp)
            vv = v(torch.tensor(np.concatenate([a['states'], a['next_states'][-1:]]), device=device)).cpu().numpy()
        mask = np.ones((len(x), 1), bool)
        mask[-1] = False
        adv, target = P.L.physical_gae(a['rewards'][:, None], vv[:-1, None], vv[1:, None],
            a['native_counts'].astype(np.int64)[:, None], mask, mask,
            gamma_native=p['gamma_native'], lambda_native=p['lambda_native'])
        batch.append(dict(x=x, raw=raw, atom=atom, old=old, adv=adv[:, 0],
                          target=loc.new_tensor(target[:, 0]), group=i//2))
        errors.append([er, lp])
    normalization = []
    for group in range(11):
        selected = [b for b in batch if b['group'] == group]
        assert len(selected) == 2
        mean = np.mean([b['adv'].mean() for b in selected])
        second = np.mean([np.mean(b['adv']**2) for b in selected])
        sd = math.sqrt(max(0, second-mean**2))+1e-8
        normalization.append(dict(group=group, mean=float(mean), sd=float(sd), episodes=2))
        for b in selected:
            b['adv'] = b['old'].new_tensor((b['adv']-mean)/sd)
    trace = []
    for k in range(p['epochs']):
        guard()
        ao.zero_grad(set_to_none=True)
        vo.zero_grad(set_to_none=True)
        pg = vl = 0.
        for b in batch:
            ratio = (P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma'])-b['old']).exp()
            loss = -torch.minimum(ratio*b['adv'], ratio.clamp(.8, 1.2)*b['adv']).mean()/len(batch)
            loss.backward()
            pg += float(loss.detach())
            loss = .5*(v(b['x'])-b['target']).square().mean()/len(batch)
            loss.backward()
            vl += float(loss.detach())
        pg_norm = float(sum(t.grad.square().sum() for t in q.parameters() if t.grad is not None).sqrt())
        aux = D.auxiliary_backward(q, groups, own, corrections, p['ppo_aux_weight'])
        an = float(torch.nn.utils.clip_grad_norm_(q.parameters(), 1., error_if_nonfinite=True))
        vn = float(torch.nn.utils.clip_grad_norm_(v.parameters(), 1., error_if_nonfinite=True))
        event('BEFORE', k)
        ao.step()
        vo.step()
        event('AFTER', k)
        assert H.frozen_digest(obj) == frozen
        with torch.no_grad():
            task_kls = []
            for group in range(11):
                task_kls.append(sum(float(((z := (P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in batch if b['group'] == group)/2)
        kl = float(np.mean(task_kls))
        trace.append(dict(epoch=k, policy_loss=pg, value_loss=vl, policy_only_gradient_norm=pg_norm,
                          auxiliary=aux, actor_norm=an, value_norm=vn, KL=kl,
                          task_KL=task_kls, maximum_task_KL=max(task_kls)))
        if kl > p['target_kl']:
            break
    return dict(trace=trace, density_errors=errors, task_advantage_normalization=normalization,
                Actor_Adam=len(trace), V_Adam=len(trace), no_BC_Adam=True,
                actor_frozen_parameter_and_Adam_hash=frozen)


def prepare():
    assert not OUT.exists(), 'Never restart'
    prior = ROOT/'stage/runs/t455_frozen_representation_ppo'
    acceptance = ROOT/'stage/runs/t456_first_ppo_acceptance'
    for folder, group in [(prior, 'T455'), (acceptance, 'T456')]:
        assert read(folder/'audit.json')['state'].startswith('PASS')
        assert read(DOC/f'{group}_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        assert all(not Path('/proc', str(pid)).exists() for pid in read(folder/'SEALED.json')['scientific_pids'])
        assert not Path('/proc', str(read(folder/'driver_started.json')['pid'])).exists()
    base = read(prior/'protocol.json')
    p = copy.deepcopy(base)
    p.pop('registration_metadata_correction', None)
    s = read(prior/'summary.json')
    p.update(id='T457_TASK_BALANCED_FROZEN_REPRESENTATION_PPO', out=str(OUT),
        sampling_initial=s['models'][0], anchor_parent=base['research_initial'],
        warmup_steps=0, steps=0, bc_steps=0, rounds=1, batch_episodes=22, task_families=11,
        seed=202611104570, max_bytes=21*2**20, full_bytes=448*1024, compact_bytes=4096, model_bytes=2*2**20,
        max_physics=9000000, max_commands=500, main_seconds=5400, learning_seconds=2700,
        audit_seconds=1200, backup_seconds=600, all_seconds=7800, preflight_commands=24,
        maximum_bc_live_Adam=0, maximum_bc_shadow_Adam=0, maximum_actual_BC_live_Adam=0, maximum_BC_shadow_Adam=0,
        maximum_ppo_live_Actor_Adam=4, maximum_ppo_live_V_Adam=4,
        maximum_ppo_shadow_Actor_Adam=4, maximum_ppo_shadow_V_Adam=4,
        maximum_new_training_episodes=22, maximum_exact_training_replays=22,
        maximum_teacher_trajectories=0, maximum_teacher_exact_replays=0,
        maximum_audit_discarded_Adam=8, preflight_discarded_Actor_Adam_upper=8, preflight_discarded_V_Adam_upper=8,
        trainable_parameter_mask_is_the_only_scientific_change_to_T453=False,
        mechanism='Existing T455 warmup checkpoint, no new BC. One22-episode own-policy batch across11 equal families, each2 original starts; per-family advantage normalization, equal episode weights. Up to4headActor+4Value PPO plus shadow.',
        new_mechanism='Explicit on-policy coverage of old8 families and wideLEFT/RIGHT plus00; do not rely only on fixed oldhistory auxiliary while sampling LEFT5 and wide.',
        training='0newBC/0newteacher,22fresh+22exact,1batch up to4PPOActor4Value and same shadow, existing130head only. Same original reward/density/auxiliary weights and fullh0.',
        comparison='Joint engineering pilot: coverage, batch size, per-family empirical normalization and one-round schedule differ from T455. Not matched sample/compute or single-factor causal attribution.',
        final_selection='One fixed final PPO proposal complete329; no panel selection, grid, extra epochs or repeat seed. Original328 successIDs and00 all required for development admission.',
        user_authorization='Nov10 user continue after full T456 found24 old regressions despite six54/54. Independent single balanced-coverage mechanism; no T455/T456 resume.',
        fresh16=None, reload78=None, six_gate=False, eight_started=False)
    prior_rows = list(map(json.loads, (prior/'cases.jsonl').read_text().splitlines()))
    query = next(r for r in prior_rows if r['role'] == 'CORRECTION_QUERY_0')
    assert query['success'] and not query['collision']
    p['correction_rows'] = [p['initial_correction'], query]
    byid = {j['id']: j for j in p['old275_jobs']+p['main_jobs']}
    families = [[byid[f'dev_task{t}_{i:02d}'] for i in [13, 26]] for t in range(8)]
    families += [[byid[f'wide117_80_dev6_{side}_{i:02d}'] for i in [13, 26]] for side in ['LEFT', 'RIGHT']]
    families.append([p['failed_job'], p['failed_job']])
    p['training_jobs'] = [j for group in families for j in group]
    p['training_family_names'] = [f'old_task{t}' for t in range(8)]+['wide_LEFT', 'wide_RIGHT', 'target_00']
    p['training_seeds'] = [p['seed']+i for i in range(22)]
    assert len(p['training_jobs']) == 22 and len(set(p['training_seeds'])) == 22
    p['historical_full_rows'] += [r for r in prior_rows if r['full']]
    p['protected_files'] = {**base['protected_files'], p['sampling_initial']['path']: p['sampling_initial']['sha256']}
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T457_PLAN.md'), str(DOC/'T457_PRIMARY_SOURCES.md'), str(DOC/'T456_FINDINGS.md'),
        str(prior/'protocol.json'), str(prior/'summary.json'), str(prior/'audit.json'), str(prior/'cases.jsonl'),
        str(acceptance/'protocol.json'), str(acceptance/'summary.json'), str(acceptance/'audit.json'),
        str(acceptance/'cases.jsonl'), p['sampling_initial']['path'], query['path']}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['constructor_reads'] = sorted(files)
    p['worst_physics'] = 2*sum(j['case']['step_cap'] for j in p['training_jobs'])+sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
    assert p['worst_physics'] < p['max_physics']
    storage = 30*p['full_bytes']+p['model_bytes']+329*p['compact_bytes']+int(4.5*2**20)
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T457_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'], storage_worst=storage,
        global_full=M.used()-M.used(OUT)+p['max_bytes'], global_before=M.used()-M.used(OUT),
        worst_physics=p['worst_physics'], max_physics=p['max_physics'],
        maximum_all_commands=413, max_commands=500, new_train=22, exact=22,
        maximum_PPO_Actor=4, maximum_PPO_Value=4, matching_shadow_each=True,
        independent_discarded_Adam_max=8, new_BC=0, new_teacher=0,
        main_seconds=5400, learning_seconds=2700, audit_seconds=1200, all_seconds=7800,
        estimated_main_minutes=[10, 20], hard_deadline_NOT_queued_duration=True,
        fresh16_reload78_NOT_reserved=True, unique_data_or_models_deleted=0))
    print('T457_REGISTERED', p['worst_physics'], flush=True)


def preflight():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime()
    a, b = objects(p), objects(p)
    before = P.state(*a)
    groups, own, corrections = auxiliary(p)
    fixture = R.synthetic(a, p)
    arrays = [copy.deepcopy(fixture[i % 4]) for i in range(22)]
    first = fit(a, arrays, p, groups, own, corrections, lambda: None, lambda *_: None)
    second = fit(b, arrays, p, groups, own, corrections, lambda: None, lambda *_: None)
    assert first == second and digest(P.state(*a)) == digest(P.state(*b))
    assert H.frozen_digest(a) == H.frozen_digest(objects(p))
    after = P.state(*a)
    buffer = io.BytesIO()
    torch.save(dict(joint_state=after, policy_hash=a[0].phash(), joint_hash=digest(after),
        architecture='plain120_128_64GRU_2', single_actor=True), buffer)
    assert len(buffer.getvalue())+65536 <= p['model_bytes']
    P.load(a, before)
    assert digest(P.state(*a)) == digest(before)
    P.load(a, after)
    assert digest(P.state(*a)) == digest(after)
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'
    folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=p['sampling_initial']['path'], job=j, seed=None, cancel_probe=True) for j in p['panel_jobs'][:8]])
        assert all(r['state'] == 'CANCEL_PASS' for r in replies) and sum(counts) == 0
    finally:
        pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24,
        discarded_Actor_Adam=first['Actor_Adam']+second['Actor_Adam'],
        discarded_Value_Adam=first['V_Adam']+second['V_Adam'], new_BC=0,
        synthetic22_fixture_NOT_navigation=True, eleven_family_pair_normalization_exact=True,
        Actor_Value_both_Adam_live_shadow_and_rollback_exact=True,
        frozen_encoder_GRU_and_mature_Adam_exact=True,
        uncompressed_mature_checkpoint_bytes=len(buffer.getvalue()), plus64KiB_within2MiB=True,
        eight_workers_zero_step_cancelled_and_stopped=True))
    print('T457_PREFLIGHT_PASS', flush=True)


def main():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime()
    E.N.offline()
    E.L.install_stop_handlers()
    rt = M.Runtime(p)
    state, error, proposal, candidate, fit_result = 'FAILED', None, None, None, None
    training_rows = []
    try:
        obj, shadow = objects(p), objects(p)
        groups, own, corrections = auxiliary(p)
        rt.pool = Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        values = rt.request(p['training_jobs'], p['sampling_initial'], 'BALANCED_TRAIN22', p['training_seeds'], full=True)
        training_rows = [r for a, r in values]
        rt.request(p['training_jobs'], p['sampling_initial'], 'BALANCED_EXACT22', p['training_seeds'], full=True, references=training_rows)
        before = P.state(*obj)
        try:
            rt.role = 'BALANCED_PPO'
            D.OUT = OUT
            live = fit(obj, [a for a, r in values], p, groups, own, corrections, lambda: rt.guard(True), D.event('balanced', 'live'))
            other = fit(shadow, [a for a, r in values], p, groups, own, corrections, lambda: rt.guard(True), D.event('balanced', 'shadow'))
            assert live == other and digest(P.state(*obj)) == digest(P.state(*shadow))
            fit_result = live
        except BaseException:
            P.load(obj, before)
            P.load(shadow, before)
            save(OUT/'interrupted_update.json', dict(Actor_Value_both_Adam_rollback_exact=digest(P.state(*obj)) == digest(before)))
            raise
        save(OUT/'fit.json', fit_result)
        proposal = P.export(obj, OUT/'proposal.pt.gz', p['model_bytes'])
        candidate, raw = M.evaluate(rt, p, proposal, 'FINAL329')
        candidate['preserves_initial328'] = set(p['initial_success_ids']) <= set(candidate['success_ids'])
        candidate['repairs00'] = p['failed_job']['id'] in candidate['success_ids']
        candidate['development_admitted'] = candidate['preserves_initial328'] and candidate['repairs00'] and len(candidate['success_ids']) == 329
        state = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, physics=sum(rt.counts), commands=rt.commands))
    finally:
        if rt.pool:
            rt.pool.close()
            rt.commands += 8
        save(OUT/'summary.json', dict(state=state, pid=os.getpid(), elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'], error=error,
            initial=p['sampling_initial'], training_rows=training_rows, fit=fit_result, proposal=proposal,
            candidate=candidate, Actor_BC_live=0, Actor_BC_shadow=0,
            PPO_Actor_live=0 if fit_result is None else fit_result['Actor_Adam'],
            PPO_Value_live=0 if fit_result is None else fit_result['V_Adam'],
            fresh16=None, reload78=None, six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=state, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN)
        rt.lock.close()
    assert state == 'COMPLETE', error
    print('T457_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json')
    verify(p, True)
    s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    ids = [s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    replays = list(map(json.loads, (OUT/'replays.jsonl').read_text().splitlines()))
    assert len(rows) == 351 and len(replays) == 22
    assert s['physics'] == sum(r['physics'] for r in rows+replays)
    assert s['commands'] == 24+16+len(rows)+len(replays) == 413
    obj = objects(p)
    frozen = H.frozen_digest(obj)
    groups, own, corrections = auxiliary(p)
    arrays = [E.N.S.arrays(r['path']) for r in s['training_rows']]
    recovered = fit(obj, arrays, p, groups, own, corrections, lambda: None, lambda *_: None)
    assert recovered == s['fit'] and digest(P.state(*obj)) == s['proposal']['joint_hash']
    assert H.frozen_digest(obj) == frozen and sha(s['proposal']['path']) == s['proposal']['sha256']
    for r in rows:
        assert sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert {k: digest(v) for k, v in a.items()} == {k: r['array_hashes'][k] for k in a}
        assert int(a['native_counts'].sum()) == r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert [bool(v) for v in a['end_flags'][-1]] == [bool(r['collision']), bool(r['success']), bool(r['timeout'])]
        if r['full']:
            assert digest(a) == r['array_digest'] and {k: digest(v) for k, v in a.items()} == r['array_hashes']
            M.audit_full_history(p, r, a)
    bypath = {r['path']: r for r in rows}
    for r in replays:
        old = bypath[r['reference_path']]
        for k in ['array_digest', 'array_hashes', 'integration_sha256']:
            assert r[k] == old[k]
    calls = list(map(json.loads, (OUT/'optimizer_calls.jsonl').read_text().splitlines()))
    for which in ['live', 'shadow']:
        assert [r[2:] for r in calls if r[:2] == ['balanced', which]] == [[w, k] for k in range(recovered['Actor_Adam']) for w in ['BEFORE', 'AFTER']]
    historic = {r['array_digest'] for r in p['historical_full_rows']}
    overlap = [dict(job_id=r['job_id'],role=r['role']) for r in s['training_rows'] if r['array_digest'] in historic]
    label_keys = []
    for row in p['correction_rows']:
        a = E.N.S.arrays(row['path'])
        for i in range(21):
            label_keys.append(digest(dict(history=a['states'][:i+1], target=a['teacher_actions'][i])))
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), new_native=0,
        discarded_historical_Actor_Adam=recovered['Actor_Adam'], discarded_historical_Value_Adam=recovered['V_Adam'],
        all_live_shadow_reconstructed_Actor_Value_both_Adam_exact=True,
        all_full120_history_hidden_density_reward_and_native_exact_replay_SHA=True,
        frozen_encoder_GRU_and_mature_Adam_exact=True, frozen_digest=frozen,
        historical_training_array_overlap=overlap, existing_early_label_entries=len(label_keys),
        existing_early_unique_complete_history_label_keys=len(set(label_keys)), labels_not_independent_navigation_samples=True,
        final_compact329_full120_NOT_reconstructed=True,
        physics=s['physics'], commands=s['commands'], fresh16=None, reload78=None, six_gate=False, eight_started=False))
    print('T457_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T457',
            '--why=Bounded task-balanced recurrent Stage PPO', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, learning_stop=now+p['learning_seconds'],
        native_stop=now+p['main_seconds'], all_stop=now+p['all_seconds'],
        protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T457_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
