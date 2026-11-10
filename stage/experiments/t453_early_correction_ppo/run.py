"""Finite early student-state correction followed by fresh recurrent PPO.

Demonstrations are auxiliary losses only. Final joint development evaluation
decides admission; the intermediate research panels never select checkpoints.
"""
from pathlib import Path
import copy, importlib.util, io, json, math, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t453_early_correction_ppo'
DOC = ROOT/'stage/docs/review_20261110'
sys.path.insert(0, str(ROOT/'stage/experiments/t451_single_failure_recovery'))
spec = importlib.util.spec_from_file_location('stage453_pinned451', ROOT/'stage/experiments/t451_single_failure_recovery/run.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
M, P, R, E, torch, np = F.M, F.P, F.R, F.E, F.torch, F.np
read, save, sha, digest = P.read, P.save, P.sha, P.digest


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes']
    assert M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


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
            rows = read(f) if f.exists() else []
            rows.append(dict(parent=os.getpid(), workers=[c.pid for c in self.children], epoch=time.time()))
            save(f, rows)
        except BaseException:
            self.close()
            raise


def correction(row):
    assert row['success'] and not row['collision'] and sha(row['path']) == row['sha256']
    a = E.N.S.arrays(row['path'])
    assert row['teacher_first'] == 20 and len(a['states']) > 21
    assert not a['ownership'][:20].any() and a['ownership'][20:].all()
    return dict(x=torch.tensor(a['states'], device='cuda'),
                y=torch.tensor(a['teacher_actions'], device='cuda', dtype=torch.float64))


def anchors(p, q):
    reference = F.anchors(q, F.reference_data(p))
    own = F.anchors(q, [dict(id=r['job_id'], task=10,
        x=(a := E.N.S.arrays(r['path']))['states'], y=a['actions']) for r in p['own_rows']])
    return reference, own


def auxiliary_backward(q, groups, own, corrections, coefficient):
    details = dict(student21=0., teacher_suffix=0., reference218=0., own10=0.)
    for g in corrections:
        loc = q.sequence(g['x'])
        early = P.action_loss(loc[:21], g['y'][:21]).mean()
        suffix = P.action_loss(loc[21:], g['y'][21:]).mean()
        (coefficient*(early+.1*suffix)/len(corrections)).backward()
        details['student21'] += float(early.detach())/len(corrections)
        details['teacher_suffix'] += float(suffix.detach())/len(corrections)
    for task in range(10):
        selected = [g for g in groups if g['task'] == task]
        weight = 1/16 if task < 8 else 1/4
        for off in range(0, len(selected), 8):
            chunk = selected[off:off+8]
            x = torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk])
            target = torch.nn.utils.rnn.pad_sequence([g['target'] for g in chunk])
            loc = q.sequence(x)
            loss = sum((loc[:len(g['x']), i]-target[:len(g['x']), i]).square().mean()
                for i, g in enumerate(chunk))*weight/len(selected)
            (coefficient*loss).backward()
            details['reference218'] += float(loss.detach())
    for off in range(0, len(own), 5):
        chunk = own[off:off+5]
        x = torch.nn.utils.rnn.pad_sequence([g['x'] for g in chunk])
        target = torch.nn.utils.rnn.pad_sequence([g['target'] for g in chunk])
        loc = q.sequence(x)
        loss = sum((loc[:len(g['x']), i]-target[:len(g['x']), i]).square().mean()
            for i, g in enumerate(chunk))/len(own)
        (coefficient*loss).backward()
        details['own10'] += float(loss.detach())
    return details


def warm_step(obj, groups, own, corrections):
    q, v, ao, vo = obj
    ao.zero_grad(set_to_none=True)
    result = auxiliary_backward(q, groups, own, corrections, 1.)
    result['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(q.parameters(), 1., error_if_nonfinite=True))
    ao.step()
    return result


def ppo(obj, arrays, p, groups, own, corrections, guard, event):
    q, v, ao, vo = obj
    device = next(q.parameters()).device
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
            assert er < 1e-10 and lp < 1e-9, (er, lp)
            vv = v(torch.tensor(np.concatenate([a['states'], a['next_states'][-1:]]), device=device)).cpu().numpy()
        mask = np.ones((len(x), 1), bool)
        mask[-1] = False
        adv, target = P.L.physical_gae(a['rewards'][:, None], vv[:-1, None], vv[1:, None],
            a['native_counts'].astype(np.int64)[:, None], mask, mask,
            gamma_native=p['gamma_native'], lambda_native=p['lambda_native'])
        batch.append(dict(x=x, raw=raw, atom=atom, old=old, adv=adv[:, 0],
                          target=loc.new_tensor(target[:, 0]), group=i//2))
        errors.append([er, lp])
    for group in [0, 1]:
        selected = [b for b in batch if b['group'] == group]
        mean = np.mean([b['adv'].mean() for b in selected])
        second = np.mean([np.mean(b['adv']**2) for b in selected])
        sd = math.sqrt(max(0, second-mean**2))+1e-8
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
            loss = -torch.minimum(ratio*b['adv'], ratio.clamp(.8, 1.2)*b['adv']).mean()/4
            loss.backward()
            pg += float(loss.detach())
            loss = .5*(v(b['x'])-b['target']).square().mean()/4
            loss.backward()
            vl += float(loss.detach())
        pg_norm = float(sum(t.grad.square().sum() for t in q.parameters() if t.grad is not None).sqrt())
        aux = auxiliary_backward(q, groups, own, corrections, p['ppo_aux_weight'])
        an = float(torch.nn.utils.clip_grad_norm_(q.parameters(), 1., error_if_nonfinite=True))
        vn = float(torch.nn.utils.clip_grad_norm_(v.parameters(), 1., error_if_nonfinite=True))
        event('BEFORE', k)
        ao.step()
        vo.step()
        event('AFTER', k)
        with torch.no_grad():
            kl = sum(float(((z := (P.L.density(q.sequence(b['x']), b['raw'], b['atom'], p['raw_sigma'])-b['old'])).exp()-1-z).mean()) for b in batch)/4
        trace.append(dict(epoch=k, policy_loss=pg, value_loss=vl, policy_only_gradient_norm=pg_norm,
                          auxiliary=aux, actor_norm=an, value_norm=vn, KL=kl))
        if kl > p['target_kl']:
            break
    return dict(trace=trace, density_errors=errors, Actor_Adam=len(trace), V_Adam=len(trace))


def event(phase, which):
    def emit(where, k):
        with (OUT/'optimizer_calls.jsonl').open('a') as f:
            f.write(json.dumps([phase, which, where, k])+'\n')
    return emit


def prepare():
    assert not OUT.exists(), 'Never restart a registered or ended directory'
    prior = ROOT/'stage/runs/t451_single_failure_recovery'
    diagnosis = ROOT/'stage/runs/t452_midstate_recovery'
    base = read(prior/'protocol.json')
    assert read(diagnosis/'audit.json')['state'] == 'PASS'
    assert read(DOC/'T452_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for directory in [prior, diagnosis]:
        assert all(not Path('/proc', str(pid)).exists() for pid in read(directory/'SEALED.json')['scientific_pids'])
        assert not Path('/proc', str(read(directory/'driver_started.json')['pid'])).exists()
    p = copy.deepcopy(base)
    for key in ['byte_references', 'constructor_reads', 'inputs', 'preserve_jobs']:
        p.pop(key, None)
    for key in list(p):
        if key.startswith('preflight_'):
            p.pop(key)
    p.update(id='T453_EARLY_STUDENT_STATE_CORRECTION_AND_FRESH_PPO', out=str(OUT),
        steps=128, bc_steps=128, warmup_steps=128, rounds=2, epochs=4, teacher_first=20,
        PPO_updates=None, V_updates=None,
        ppo_aux_weight=.1, seed=202611104530,
        max_bytes=20*2**20, model_bytes=2*2**20, full_bytes=448*1024, compact_bytes=4096,
        max_physics=9000000, max_commands=500, main_seconds=5400, learning_seconds=2700,
        audit_seconds=1200, backup_seconds=600, all_seconds=7800, preflight_commands=24,
        maximum_bc_live_Adam=128, maximum_bc_shadow_Adam=128,
        maximum_actual_BC_live_Adam=128, maximum_BC_shadow_Adam=128,
        maximum_ppo_live_Actor_Adam=8, maximum_ppo_live_V_Adam=8,
        maximum_ppo_shadow_Actor_Adam=8, maximum_ppo_shadow_V_Adam=8,
        maximum_new_training_episodes=8, maximum_exact_training_replays=8,
        maximum_teacher_trajectories=2, maximum_teacher_exact_replays=2,
        maximum_audit_discarded_Adam=144, preflight_discarded_Actor_Adam_upper=12,
        preflight_discarded_V_Adam_upper=8,
        fresh16=None, reload78=None, six_gate=False, eight_started=False,
        mechanism='128 maskedBC warmup(student-induced states0..20 weight1, teacher suffix21.. weight.1)+2round4fresh recurrentPPO, all218 current-policy anchors and10own histories; auxiliaryPPO weight.1, no BC-fit entry gate.',
        new_mechanism='Weight student-induced first21 state labels separately from later teacher-induced suffix, then execute fresh student PPO with separately logged auxiliary retention.',
        training='Joint finite pilot, all52866 Actor parameters; mature ActorAdam and parent independentV. Standard DAgger-inspired masking/PPO not original.',
        comparison='No single-module causal control: changed coverage weighting and actual PPO execution. No claim128steps or8episodes sufficient.',
        final_selection='Fixed last research model, no intermediate best-panel selection or semantic rollback. Research intermediate regressions measured, final original328 must remain and00 must repair for development admission.',
        user_authorization='Nov10 user explicitly starts next round after T452 recovery domain audit. Independent finite learning pilot, not extension of T451 BC or T452 diagnosis.',
        complete_fresh16_reload78_NOT_RESERVED=True, no_runtime_teacher_or_router=True)
    rows = [json.loads(x) for x in (diagnosis/'cases.jsonl').read_text().splitlines()]
    p['initial_correction'] = next(r for r in rows if r['teacher'] and r['teacher_first'] == 20)
    assert p['initial_correction']['success'] and not p['initial_correction']['collision']
    p['own_rows'] = [json.loads(x) for x in (prior/'cases.jsonl').read_text().splitlines() if json.loads(x)['role'] == 'INITIAL_OWN10']
    assert len(p['own_rows']) == 10 and all(r['success'] for r in p['own_rows'])
    p['panel_jobs'] = base['preserve_jobs']+[p['failed_job']]
    byid = {j['id']: j for j in p['old275_jobs']+p['main_jobs']}
    p['training_jobs'] = [[byid[n] for n in names] for names in [
        ['dev_task6_00', 'dev_task6_03', 'wide117_80_dev6_LEFT_00', 'wide117_80_dev6_RIGHT_00'],
        ['dev_task6_00', 'dev_task6_24', 'wide117_80_dev6_LEFT_13', 'wide117_80_dev6_RIGHT_13']]]
    p['historical_full_rows'] = []
    for name in ['t449_offline_joint_recovery', 't451_single_failure_recovery', 't452_midstate_recovery']:
        p['historical_full_rows'] += [r for r in map(json.loads, (ROOT/'stage/runs'/name/'cases.jsonl').read_text().splitlines()) if r.get('full')]
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T453_PLAN.md'), str(DOC/'T453_PRIMARY_SOURCES.md'), str(diagnosis/'audit.json'), str(diagnosis/'summary.json'), str(diagnosis/'cases.jsonl'), p['initial_correction']['path']}
    files |= {r['path'] for r in p['own_rows']}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['constructor_reads'] = sorted(files)
    cap = lambda jobs: sum(j['case']['step_cap'] for j in jobs)
    p['worst_physics'] = 4*cap([p['failed_job']])+2*sum(cap(js) for js in p['training_jobs'])+3*cap(p['panel_jobs'])+cap(p['main_jobs']+p['old275_jobs'])
    assert p['worst_physics'] < p['max_physics']
    storage = 18*p['full_bytes']+3*p['model_bytes']+362*p['compact_bytes']+3*2**20
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] <= 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T453_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'],
        storage_worst=storage, full10plus8partial=18*p['full_bytes'], models3=3*p['model_bytes'],
        global_full=M.used()-M.used(OUT)+p['max_bytes'], global_before=M.used()-M.used(OUT),
        worst_physics=p['worst_physics'], max_physics=p['max_physics'],
        maximum_all_commands=422, max_commands=500, main_seconds=5400, learning_seconds=2700,
        audit_seconds=1200, all_seconds=7800, estimated_main_minutes=[10, 25],
        BC_live=128, BC_shadow=128, PPO_Actor_live_max=8, PPO_Value_live_max=8,
        PPO_Actor_shadow_max=8, PPO_Value_shadow_max=8, independent_discarded_Adam_max=144,
        fresh16_reload78_NOT_reserved=True, unique_bytes_deleted=0))
    print('T453_REGISTERED', p['worst_physics'], flush=True)


def preflight():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime()
    obj = F.objects(p)
    shadow = F.objects(p)
    before = P.state(*obj)
    groups, own = anchors(p, obj[0])
    corrections = [correction(p['initial_correction'])]
    start = time.monotonic()
    warm = []
    for _ in range(2):
        a = warm_step(obj, groups, own, corrections)
        b = warm_step(shadow, groups, own, corrections)
        assert a == b
        warm.append(a)
    torch.cuda.synchronize()
    estimate = (time.monotonic()-start)/2*p['warmup_steps']
    assert estimate < 1200 and digest(P.state(*obj)) == digest(P.state(*shadow))
    assert warm[0]['reference218'] < 1e-24 and warm[0]['own10'] < 1e-24
    arrays = R.synthetic(obj, p)
    a = ppo(obj, arrays, p, groups, own, corrections, lambda: None, lambda *_: None)
    b = ppo(shadow, arrays, p, groups, own, corrections, lambda: None, lambda *_: None)
    assert a == b and digest(P.state(*obj)) == digest(P.state(*shadow))
    after = P.state(*obj)
    mem = io.BytesIO()
    torch.save(dict(joint_state=after, policy_hash=obj[0].phash(), joint_hash=digest(after),
                    architecture='plain120_128_64GRU_2', single_actor=True), mem)
    assert len(mem.getvalue())+65536 <= p['model_bytes']
    P.load(obj, before)
    assert digest(P.state(*obj)) == digest(before)
    P.load(obj, after)
    assert digest(P.state(*obj)) == digest(after)
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'
    folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=p['research_initial']['path'], job=j, seed=None, cancel_probe=True) for j in p['panel_jobs'][:8]])
        assert all(r['state'] == 'CANCEL_PASS' for r in replies) and sum(counts) == 0
    finally:
        pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24,
        discarded_Actor_Adam=4+a['Actor_Adam']+b['Actor_Adam'],
        discarded_V_Adam=a['V_Adam']+b['V_Adam'], synthetic_PPO_NOT_navigation=True,
        full_history_live_shadow_and_two_Adam_rollback_exact=True,
        initial_anchor_losses=warm[0], estimated128_BC_live_shadow_seconds=estimate,
        mature_twoAdam_uncompressed_bytes=len(mem.getvalue()), plus64KiB_within2MiB=True,
        eight_native_workers_cancelled_and_stopped=True))
    print('T453_PREFLIGHT_PASS', estimate, flush=True)


def main():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    setup_runtime()
    E.N.offline()
    E.L.install_stop_handlers()
    rt = M.Runtime(p)
    state, error, candidate = 'FAILED', None, None
    rounds, trace, models = [], [], []
    try:
        obj, shadow = F.objects(p), F.objects(p)
        groups, own = anchors(p, obj[0])
        correction_rows = [p['initial_correction']]
        corrections = [correction(r) for r in correction_rows]
        for k in range(p['warmup_steps']):
            rt.role = 'MASKED_EARLY_WARMUP'
            rt.guard(True)
            event('warmup', 'live')('BEFORE', k)
            a = warm_step(obj, groups, own, corrections)
            event('warmup', 'live')('AFTER', k)
            event('warmup', 'shadow')('BEFORE', k)
            b = warm_step(shadow, groups, own, corrections)
            event('warmup', 'shadow')('AFTER', k)
            assert a == b
            trace.append(a)
            if k % 16 == 0:
                assert digest(P.state(*obj)) == digest(P.state(*shadow))
                save(OUT/'fit_progress.json', dict(phase='warmup', steps=k+1, epoch=time.time(), losses=a))
        save(OUT/'warmup_trace.json', trace)
        model = P.export(obj, OUT/'warmup.pt.gz', p['model_bytes'])
        models.append(model)
        rt.pool = Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        rt.request(p['panel_jobs'], model, 'WARMUP_PANEL')
        for rnd, jobs in enumerate(p['training_jobs']):
            rt.guard(True)
            query = rt.request([p['failed_job']], model, f'CORRECTION_QUERY_{rnd}', full=True, query=True)
            rt.request([p['failed_job']], model, f'CORRECTION_QUERY_EXACT_{rnd}', full=True,
                       references=[r for a, r in query], query=True)
            if query[0][1]['success'] and not query[0][1]['collision']:
                correction_rows.append(query[0][1])
                corrections.append(correction(query[0][1]))
            seeds = [p['seed']+rnd*10+i for i in range(4)]
            values = rt.request(jobs, model, f'PPO_TRAIN_{rnd}', seeds, full=True)
            rt.request(jobs, model, f'PPO_EXACT_{rnd}', seeds, full=True, references=[r for a, r in values])
            before = P.state(*obj)
            try:
                a = ppo(obj, [a for a, r in values], p, groups, own, corrections,
                        lambda: rt.guard(True), event(f'ppo{rnd}', 'live'))
                b = ppo(shadow, [a for a, r in values], p, groups, own, corrections,
                        lambda: rt.guard(True), event(f'ppo{rnd}', 'shadow'))
                assert a == b and digest(P.state(*obj)) == digest(P.state(*shadow))
            except BaseException:
                P.load(obj, before)
                P.load(shadow, before)
                save(OUT/f'interrupted_round_{rnd}.json', dict(Actor_Value_and_both_Adam_rollback_exact=digest(P.state(*obj)) == digest(before)))
                raise
            model = P.export(obj, OUT/f'round_{rnd}.pt.gz', p['model_bytes'])
            models.append(model)
            panel = rt.request(p['panel_jobs'], model, f'PPO_PANEL_{rnd}')
            rounds.append(dict(round=rnd, model=model, fit=a, corrections=copy.deepcopy(correction_rows),
                train_rows=[r for a, r in values], panel_success_ids=M.safe_ids(panel),
                retained_for_research_NOT_promoted=True))
            save(OUT/'rounds.json', rounds)
        candidate, raw = M.evaluate(rt, p, model, 'FINAL329')
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
            warmup_steps_completed=len(trace), models=models, rounds=rounds, candidate=candidate,
            Actor_BC_live=len(trace), Actor_BC_shadow=len(trace),
            PPO_Actor_live=sum(r['fit']['Actor_Adam'] for r in rounds),
            PPO_Value_live=sum(r['fit']['V_Adam'] for r in rounds),
            fresh16=None, reload78=None, six_gate=False, eight_started=False,
            all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=state, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN)
        rt.lock.close()
    assert state == 'COMPLETE', error
    print('T453_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json')
    verify(p, True)
    s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    ids = [s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    replays = list(map(json.loads, (OUT/'replays.jsonl').read_text().splitlines()))
    assert s['physics'] == sum(r['physics'] for r in rows+replays)
    assert s['commands'] == 24+16+len(rows)+len(replays)
    obj = F.objects(p)
    groups, own = anchors(p, obj[0])
    correction_rows = [p['initial_correction']]
    corrections = [correction(r) for r in correction_rows]
    trace = [warm_step(obj, groups, own, corrections) for _ in range(p['warmup_steps'])]
    assert trace == read(OUT/'warmup_trace.json')
    assert digest(P.state(*obj)) == s['models'][0]['joint_hash']
    calls = p['warmup_steps']
    for rnd, record in enumerate(s['rounds']):
        correction_rows = record['corrections']
        corrections = [correction(r) for r in correction_rows]
        arrays = [E.N.S.arrays(r['path']) for r in record['train_rows']]
        result = ppo(obj, arrays, p, groups, own, corrections, lambda: None, lambda *_: None)
        assert result == record['fit'] and digest(P.state(*obj)) == record['model']['joint_hash']
        calls += result['Actor_Adam']+result['V_Adam']
    expert = E.safe_initial(E.N.science(p), p).eval().requires_grad_(False)
    for r in rows:
        assert sha(r['path']) == r['sha256']
        a = E.N.S.arrays(r['path'])
        assert int(a['native_counts'].sum()) == r['physics']
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        if r['full']:
            assert digest(a) == r['array_digest'] and {k: digest(v) for k, v in a.items()} == r['array_hashes']
            M.audit_full_history(p, r, a)
            if r['teacher']:
                with torch.no_grad():
                    labels = expert.base_actions(torch.tensor(a['states'])).float().numpy()
                assert np.array_equal(labels, a['teacher_actions'])
    bypath = {r['path']: r for r in rows}
    for r in replays:
        old = bypath[r['reference_path']]
        assert r['array_digest'] == old['array_digest'] and r['integration_sha256'] == old['integration_sha256']
    ev = list(map(json.loads, (OUT/'optimizer_calls.jsonl').read_text().splitlines()))
    for phase, n in [('warmup', p['warmup_steps'])]+[(f'ppo{i}', r['fit']['Actor_Adam']) for i, r in enumerate(s['rounds'])]:
        for which in ['live', 'shadow']:
            assert [r[2:] for r in ev if r[:2] == [phase, which]] == [[w, k] for k in range(n) for w in ['BEFORE', 'AFTER']]
    historic = {r['array_digest'] for r in p['historical_full_rows']}
    overlap = [dict(role=r['role'], job_id=r['job_id']) for r in rows if r['full'] and r['array_digest'] in historic]
    assert all(sha(m['path']) == m['sha256'] for m in s['models'])
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), new_native=0,
        discarded_historical_Adam=calls, original_live_shadow_and_reconstructed_all_Actor_Value_Adam_exact=True,
        full_training_history_hidden_density_teacher_reward_and_SHA_exact=True,
        compact_final329_full120_NOT_reconstructed=True, historical_full_array_overlap=overlap,
        physics=s['physics'], commands=s['commands'], candidate=s['candidate'],
        fresh16=None, reload78=None, six_gate=False, eight_started=False))
    print('T453_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T453',
            '--why=Bounded early student-state correction and fresh PPO', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, learning_stop=now+p['learning_seconds'],
        native_stop=now+p['main_seconds'], all_stop=now+p['all_seconds'],
        protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T453_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'entry_error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time(), native_count_not_inferred=True))
        raise
