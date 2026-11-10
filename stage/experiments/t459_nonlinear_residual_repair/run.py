"""Finite supervised repair through one new internal neural residual head.

The complete original328 Actor is frozen. Only1074 added parameters learn.
No new RL, teacher queries, runtime switch, planner or waypoint is introduced.
"""
from pathlib import Path
import copy, gzip, importlib.util, io, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t459_nonlinear_residual_repair'
DOC = ROOT/'stage/docs/review_20261110'
spec = importlib.util.spec_from_file_location('stage459_pinned453', ROOT/'stage/experiments/t453_early_correction_ppo/run.py')
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)
F, M, P, R, E, torch, np = D.F, D.M, D.P, D.R, D.E, D.torch, D.np
read, save, sha, digest = P.read, P.save, P.sha, P.digest
ORIGINAL_FROZEN = P.frozen
ARCH = 'plain120_128_GRU64_frozen_base_plus_internal_MLP64_16_2_zero_initialized'


class ResidualHead(torch.nn.Module):
    def __init__(self, base, seed):
        super().__init__()
        self.base = base.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.residual = torch.nn.Sequential(torch.nn.Linear(64, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)).double()
            torch.nn.init.zeros_(self.residual[-1].weight)
            torch.nn.init.zeros_(self.residual[-1].bias)

    def forward(self, hidden):
        # Both neural terms execute for every observation; no task-ID routing.
        return self.base(hidden)+self.residual(hidden)


class Actor(P.Actor):
    def __init__(self, p):
        super().__init__(p)
        self.load_state_dict(P.checkpoint(p['research_initial']['path'])['joint_state']['actor'])
        self.requires_grad_(False)
        self.head = ResidualHead(self.head, p['residual_seed'])

    def phash(self):
        return digest(dict(architecture=ARCH, state=self.state_dict(), sigma=self.p['raw_sigma']))


def base_digest(q):
    state = {}
    for name, value in q.state_dict().items():
        if name.startswith('head.residual.'):
            continue
        state[name.replace('head.base.', 'head.')] = value
    return digest(state)


def make(p, device='cuda'):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    q = Actor(p).to(device)
    # Frozen recurrent features require no cuDNN recurrent backward path.
    q.core.eval()
    opt = torch.optim.Adam(q.head.residual.parameters(), lr=p['repair_lr'], foreach=False, fused=False)
    assert sum(v.numel() for v in q.parameters() if v.requires_grad) == 1074
    assert sum(v.numel() for v in q.parameters() if not v.requires_grad) == 52866
    assert base_digest(q) == p['base_actor_digest']
    return q, opt


def state(q, opt):
    return P.L.cpu(dict(actor=q.state_dict(), residual_opt=opt.state_dict()))


def load(obj, saved):
    obj[0].load_state_dict(saved['actor'])
    obj[1].load_state_dict(copy.deepcopy(saved['residual_opt']))


def export(obj, path, limit):
    s = state(*obj)
    mem = io.BytesIO()
    torch.save(dict(residual_state=s, architecture=ARCH, policy_hash=obj[0].phash(),
        joint_hash=digest(s), single_actor=True, frozen_parent=obj[0].p['research_initial']), mem)
    assert len(mem.getvalue())+65536 <= limit
    blob = gzip.compress(mem.getvalue(), mtime=0)
    assert len(blob) <= limit and not Path(path).exists()
    Path(path).write_bytes(blob)
    return dict(path=str(path), sha256=sha(path), policy_hash=obj[0].phash(), joint_hash=digest(s),
                bytes=len(blob), uncompressed_bytes=len(mem.getvalue()))


def frozen(p, path):
    c = P.checkpoint(path)
    if c.get('architecture') != ARCH:
        return ORIGINAL_FROZEN(p, path)
    assert c['frozen_parent'] == p['research_initial']
    assert digest(c['residual_state']) == c['joint_hash']
    q = Actor(p).eval().requires_grad_(False)
    q.load_state_dict(c['residual_state']['actor'])
    assert q.phash() == c['policy_hash'] and base_digest(q) == p['base_actor_digest']
    return q


def verify(p, full=False):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    assert M.used(OUT) <= p['max_bytes'] and M.used()-M.used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30
    for f, h in (p['inputs'] if full else p['protected_files']).items():
        assert sha(f) == h, f


def worker(pipe, index, p, counts):
    P.frozen = frozen
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


def setup_runtime():
    M.OUT, M.verify, M.Pool = OUT, verify, Pool
    M.setup_runtime()


def auxiliary_data(p):
    parent = ORIGINAL_FROZEN(p, p['research_initial']['path']).cuda()
    groups, own = D.anchors(p, parent)
    correction = D.correction(p['initial_correction'])
    return groups, own, [correction]


def step(obj, data):
    q, opt = obj
    opt.zero_grad(set_to_none=True)
    result = D.auxiliary_backward(q, *data, 1.)
    result['gradient_norm'] = float(torch.nn.utils.clip_grad_norm_(q.head.residual.parameters(), 1., error_if_nonfinite=True))
    assert all(v.grad is None for name, v in q.named_parameters() if not name.startswith('head.residual.'))
    opt.step()
    assert base_digest(q) == q.p['base_actor_digest']
    return result


def prepare():
    assert not OUT.exists(), 'Never restart'
    prior = ROOT/'stage/runs/t458_warmup_regression_control'
    assert read(prior/'audit.json')['state'] == 'PASS_FROZEN_COMPACT_EVIDENCE'
    assert read(DOC/'T458_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert all(not Path('/proc', str(pid)).exists() for pid in read(prior/'SEALED.json')['scientific_pids'])
    assert not Path('/proc', str(read(prior/'driver_started.json')['pid'])).exists()
    base = read(ROOT/'stage/runs/t455_frozen_representation_ppo/protocol.json')
    p = copy.deepcopy(base)
    p.pop('registration_metadata_correction', None)
    p.pop('trainable_parameter_mask_is_the_only_scientific_change_to_T453', None)
    p.update(id='T459_INTERNAL_NONLINEAR_RESIDUAL_SUPERVISED_REPAIR', out=str(OUT),
        architecture=ARCH, actor_trainable_parameters=1074, actor_frozen_parameters=52866,
        trainable_prefixes=['head.residual.'], frozen_prefixes=['encoder.', 'core.', 'head.base.'],
        mechanism='Original328 entire Actor frozen; add zero-initialized64-16Tanh-2 internal logit residual, fresh residual-onlyAdam,128existing-correction BC+128shadow; fixed final329. No new teacher, PPO or Value training.',
        new_mechanism='Test nonlinear context-specific correction while retaining the complete original recurrent base; zero initial residual is an initial identity, not final retention guarantee.',
        training='128supervised residualAdam+128shadow;0newPPO/Value/native training/teacher queries.218reference+10own function anchors and one existing safe4sec recovery, early21 andsuffix.1 unchanged.',
        comparison='Capacity and fresh optimizer jointly change versus T455 warmup; single-seed engineering pilot, not isolated capacity/freshAdam causality or originality.',
        user_authorization='Nov10 user inspect and think next; continuing delegated Stage work after T458 identified warmup and PPO regression, register one finite new neural residual mechanism, not extending prior updates.',
        warmup_steps=128, rounds=0, epochs=0, training_jobs=[], repair_lr=3e-5, actor_lr=3e-5,
        residual_seed=202611104590, raw_sigma=base['raw_sigma'],
        max_bytes=10*2**20, model_bytes=768*1024, full_bytes=448*1024, compact_bytes=4096,
        max_physics=7000000, max_commands=400, preflight_commands=24,
        main_seconds=2700, learning_seconds=1200, audit_seconds=600, backup_seconds=600, all_seconds=4500,
        maximum_bc_live_Adam=128, maximum_bc_shadow_Adam=128,
        maximum_actual_BC_live_Adam=128, maximum_BC_shadow_Adam=128,
        maximum_ppo_live_Actor_Adam=0, maximum_ppo_live_V_Adam=0, PPO_updates=0, V_updates=0,
        maximum_new_training_episodes=0, maximum_exact_training_replays=0,
        maximum_teacher_trajectories=0, maximum_teacher_exact_replays=0,
        maximum_audit_discarded_Adam=128, preflight_discarded_Actor_Adam=8, preflight_discarded_V_Adam=0,
        zero_model_path=str(OUT/'initial_zero.pt.gz'),
        final_selection='Fixed128th proposal full329; original328 successIDs preserved and00 repaired required for development admission. No intermediate choice.',
        six_gate=False, fresh16=None, reload78=None, eight_started=False,
        original_parent_and_rejected_runs_unchanged=True, complete_fresh16_reload78_NOT_RESERVED=True)
    p['base_actor_digest'] = digest(P.checkpoint(p['research_initial']['path'])['joint_state']['actor'])
    assert p['initial_correction']['teacher_first'] == 20 and p['initial_correction']['success']
    p['worst_physics'] = sum(j['case']['step_cap'] for j in p['main_jobs']+p['old275_jobs'])
    assert p['worst_physics'] < p['max_physics']
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T459_PLAN.md'), str(DOC/'T459_PRIMARY_SOURCES.md'), str(DOC/'T458_FINDINGS_AND_NEXT.md'),
        str(prior/'protocol.json'), str(prior/'summary.json'), str(prior/'audit.json'),
        str(prior/'stage_comparison.json'), str(DOC/'T458_FINAL_REPORT.json'),
        str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T458.json')}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    p['constructor_reads'] = sorted(files)
    storage = 329*p['compact_bytes']+8*p['full_bytes']+2*p['model_bytes']+3*2**20
    assert storage <= p['max_bytes'] and M.used()+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p, True)
    save(DOC/'T459_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'],
        storage_worst=storage, global_full=M.used()-M.used(OUT)+p['max_bytes'],
        max_physics=p['max_physics'], worst_physics=p['worst_physics'], maximum_all_commands=369,
        main_seconds=2700, learning_seconds=1200, audit_seconds=600, all_seconds=4500,
        estimated_main_minutes=[10, 25], hard_cap_NOT_queued_duration=True,
        BC_live=128, BC_shadow=128, new_native_training=0, new_PPO=0, new_teacher_queries=0,
        copied_parent_models=0, unique_bytes_deleted=0, fresh16_reload78_NOT_reserved=True))
    print('T459_REGISTERED', flush=True)


def preflight():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert not (OUT/'preflight.json').exists()
    setup_runtime()
    a, b = make(p), make(p)
    initial = state(*a)
    data = auxiliary_data(p)
    # Check the full recorded histories, not merely individual sensor packets.
    with torch.no_grad():
        for g in data[0]+data[1]:
            assert torch.equal(a[0].sequence(g['x']), g['target'])
        x = data[2][0]['x'][:64]
        parent = ORIGINAL_FROZEN(p, p['research_initial']['path']).cuda()
        assert torch.equal(a[0].sequence(x), parent.sequence(x))
    qcpu = Actor(p).eval()
    basecpu = ORIGINAL_FROZEN(p, p['research_initial']['path'])
    xx = x.cpu()
    h1 = h2 = None
    with torch.no_grad():
        for row in xx:
            o1, l1, h1, _ = qcpu.step(row[None], h1)
            o2, l2, h2, _ = basecpu.step(row[None], h2)
            assert torch.equal(o1, o2) and torch.equal(l1, l2) and torch.equal(h1, h2)
        step_locations = torch.as_tensor(np.stack([t['locations'] for t in qcpu.tape]))
        assert torch.allclose(qcpu.sequence(xx), step_locations, atol=1e-10, rtol=0)
    # Four matched synthetic steps per clone, discarded and fully rolled back.
    target = a[0].sequence(x).detach()+.01
    for _ in range(4):
        losses = []
        for q, opt in [a, b]:
            opt.zero_grad(set_to_none=True)
            loss = (q.sequence(x)-target).square().mean()
            loss.backward()
            opt.step()
            assert base_digest(q) == p['base_actor_digest']
            losses.append(float(loss.detach()))
        assert losses[0] == losses[1] and digest(state(*a)) == digest(state(*b))
    mem = io.BytesIO()
    torch.save(dict(residual_state=state(*a), architecture=ARCH), mem)
    assert len(mem.getvalue())+65536 <= p['model_bytes']
    load(a, initial)
    load(b, initial)
    assert digest(state(*a)) == digest(initial) == digest(state(*b))
    zero = export(a, p['zero_model_path'], p['model_bytes'])
    assert frozen(p, zero['path']).phash() == a[0].phash()
    counts = R.M.mp.get_context('spawn').RawArray('q', 9)
    folder = OUT/'preflight_native'
    folder.mkdir()
    pool = Pool(dict(p, out=str(folder), native_stop_epoch=time.time()+180, max_physics=1), counts, lambda: None)
    try:
        replies = pool.request([dict(model=zero['path'], job=j, seed=None, cancel_probe=True) for j in p['old8_jobs']])
        assert all(r['state'] == 'CANCEL_PASS' for r in replies) and sum(counts) == 0
    finally:
        pool.close()
    save(OUT/'preflight.json', dict(state='PASS', native=0, commands=24,
        discarded_synthetic_Adam=8, no_new_PPO_or_teacher=True, zero_model=zero,
        Actor_trainable=1074, Actor_frozen=52866, original_parent_full_history_location_exact=True,
        real_hidden_carry_and_CPU_actions_exact_initial_parent=True,
        synthetic_live_shadow_and_full_Actor_residualAdam_rollback_exact=True,
        uncompressed_mature_residual_checkpoint_bytes=len(mem.getvalue()),
        eight_workers_zero_step_cancelled_and_stopped=True))
    print('T459_PREFLIGHT_PASS', flush=True)


def main():
    p = read(OUT/'protocol.json')
    verify(p, True)
    pre = read(OUT/'preflight.json')
    assert pre['state'] == 'PASS' and not (OUT/'summary.json').exists()
    assert sha(pre['zero_model']['path']) == pre['zero_model']['sha256']
    setup_runtime()
    E.N.offline()
    E.L.install_stop_handlers()
    rt = M.Runtime(p)
    status, error, candidate, proposal = 'FAILED', None, None, None
    completed = 0
    obj = shadow = initial = None
    try:
        obj, shadow = make(p), make(p)
        initial = state(*obj)
        assert obj[0].phash() == pre['zero_model']['policy_hash']
        data = auxiliary_data(p)
        trace = []
        for k in range(p['warmup_steps']):
            rt.role = 'NONLINEAR_RESIDUAL_BC'
            rt.guard(True)
            values = []
            for which, objects in [('live', obj), ('shadow', shadow)]:
                M.events('residual_BC', which)('BEFORE', k)
                values.append(step(objects, data))
                M.events('residual_BC', which)('AFTER', k)
            assert values[0] == values[1] and digest(state(*obj)) == digest(state(*shadow))
            completed += 1
            trace.append(values[0])
            if k % 16 == 0:
                save(OUT/'fit_progress.json', dict(steps=completed, latest=values[0], epoch=time.time()))
        save(OUT/'repair_trace.json', trace)
        proposal = export(obj, OUT/'proposal.pt.gz', p['model_bytes'])
        rt.pool = Pool(dict(p, native_stop_epoch=rt.deadline), rt.counts, rt.guard)
        candidate, _ = M.evaluate(rt, p, proposal, 'RESIDUAL_FINAL329')
        candidate['preserves_initial328'] = set(p['initial_success_ids']) <= set(candidate['success_ids'])
        candidate['repairs00'] = p['failed_job']['id'] in candidate['success_ids']
        candidate['development_admitted'] = len(candidate['success_ids']) == 329
        status = 'COMPLETE'
    except BaseException:
        error = traceback.format_exc()
        if obj is not None and initial is not None:
            load(obj, initial)
            if shadow is not None:
                load(shadow, initial)
            assert digest(state(*obj)) == digest(initial)
        save(OUT/'failure.json', dict(error=error, completed_paired_BC=completed,
            Actor_and_residualAdam_rolled_back=initial is not None, raw_and_failed_proposal_preserved=True))
    finally:
        if rt.pool:
            rt.pool.close()
            rt.commands += 8
        events = list(map(json.loads, (OUT/'optimizer_calls.jsonl').read_text().splitlines())) if (OUT/'optimizer_calls.jsonl').exists() else []
        actual = {which: sum(r[1:3] == [which, 'AFTER'] for r in events) for which in ['live', 'shadow']}
        save(OUT/'summary.json', dict(state=status, pid=os.getpid(), elapsed_s=time.time()-rt.start,
            physics=sum(rt.counts), commands=rt.commands+p['preflight_commands'], error=error,
            proposal=proposal, candidate=candidate, Actor_BC_live=actual['live'], Actor_BC_shadow=actual['shadow'],
            completed_paired_BC=completed, interrupted_last_BEFORE_without_AFTER_not_assumed_complete=status != 'COMPLETE',
            PPO_Actor_live=0, PPO_Value_live=0, new_teacher=0, new_native_training=0,
            original_parent_model=p['research_initial'], frozen_original_base=True,
            retained_model=p['research_initial'], research_proposal_not_promoted=True,
            fresh16=None, reload78=None, six_gate=False, eight_started=False, all_native_workers_stopped=True))
        save(OUT/'status.json', dict(state=status, epoch=time.time()))
        R.M.fcntl.flock(rt.lock, R.M.fcntl.LOCK_UN)
        rt.lock.close()
    assert status == 'COMPLETE', error
    print('T459_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json')
    verify(p, True)
    s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not (OUT/'audit.json').exists()
    ids = [s['pid']]+[pid for g in read(OUT/'worker_generations.json') for pid in g['workers']]
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    obj = make(p)
    data = auxiliary_data(p)
    trace = [step(obj, data) for _ in range(p['warmup_steps'])]
    assert trace == read(OUT/'repair_trace.json')
    assert digest(state(*obj)) == s['proposal']['joint_hash'] and obj[0].phash() == s['proposal']['policy_hash']
    assert base_digest(obj[0]) == p['base_actor_digest']
    assert sha(s['proposal']['path']) == s['proposal']['sha256']
    assert frozen(p, s['proposal']['path']).phash() == obj[0].phash()
    events = list(map(json.loads, (OUT/'optimizer_calls.jsonl').read_text().splitlines()))
    for which in ['live', 'shadow']:
        assert [r[2:] for r in events if r[:2] == ['residual_BC', which]] == [[where, k] for k in range(128) for where in ['BEFORE', 'AFTER']]
    assert len(events) == 512
    rows = list(map(json.loads, (OUT/'cases.jsonl').read_text().splitlines()))
    assert len(rows) == len({r['job_id'] for r in rows}) == 329
    assert s['physics'] == sum(r['physics'] for r in rows) and s['commands'] == 24+16+len(rows)
    for r in rows:
        assert sha(r['path']) == r['sha256'] and r['role'] == 'RESIDUAL_FINAL329'
        a = E.N.S.arrays(r['path'])
        assert {k: digest(v) for k, v in a.items()} == {k: r['array_hashes'][k] for k in a}
        assert r['loaded_model_sha256'] == s['proposal']['sha256'] and r['policy_hash'] == obj[0].phash()
        assert r['seed'] is None and not r['teacher'] and not r['administrative_censor']
        assert int(a['native_counts'].sum()) == r['physics'] and np.abs(a['actions']).max() <= 1
        assert not a['end_flags'][:-1].any() and a['end_flags'][-1].sum() == 1
        assert [bool(v) for v in a['end_flags'][-1]] == [r['collision'], r['success'], r['timeout']]
    assert R.M.tally(rows[:54]) == s['candidate']['wide54'] and R.M.tally(rows[54:]) == s['candidate']['old275']
    assert digest(P.checkpoint(p['research_initial']['path'])['joint_state']) == p['research_initial']['joint_hash']
    save(OUT/'audit.json', dict(state='PASS', pid=os.getpid(), new_native=0,
        discarded_historical_residual_Adam=128, full_Actor_residualAdam_live_shadow_reconstructed_exact=True,
        entire_original52866parameters_and_parent_ActorValueAdam_file_unchanged=True,
        base_actor_digest=p['base_actor_digest'], same_proposal_all329=True,
        compact_actions_termination_native_counts_SHA_verified=True,
        full120_hidden_history_of329_NOT_reconstructed=True, physics=s['physics'], commands=s['commands'],
        existing_safe_mixed_recovery_NOT_new_onpolicy=True, new_PPO=0, new_teacher=0,
        fresh16=None, reload78=None, six_gate=False, eight_started=False))
    print('T459_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p, True)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    src = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert src['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage-T459',
            '--why=Bounded internal neural residual Stage repair', 'timeout', '--signal=TERM', '--kill-after=2',
            str(p['all_seconds']), 'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, learning_stop=now+p['learning_seconds'],
        native_stop=now+p['main_seconds'], all_stop=now+p['all_seconds'],
        protocol_sha256=sha(OUT/'protocol.json'), source_remote=src['remote_commit']))
    print('T459_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time()))
        raise
