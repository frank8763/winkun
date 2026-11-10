"""Exact descriptive frozen function audit on identical real histories.

No simulator/environment construction or learning. Mixed parameter blocks are
evaluated only offline; they are never exported or used to control a vehicle.
"""
from pathlib import Path
import copy, importlib.util, json, os, shutil, subprocess, sys, time, traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t454_frozen_function_audit'
DOC = ROOT/'stage/docs/review_20261110'
sys.path.insert(0, str(ROOT/'stage/experiments/t446_plain_joint'))
import t446_learning as P
torch, np, E = P.torch, P.np, P.E
read, save, sha, digest = P.read, P.save, P.sha, P.digest


def used(folder=None):
    return sum(f.stat().st_size for f in (Path(folder) if folder else ROOT/'stage/runs').rglob('*') if f.is_file())


def verify(p):
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    for f, h in p['inputs'].items():
        assert sha(f) == h, f
    assert used(OUT) <= p['max_bytes'] and used()-used(OUT)+p['max_bytes'] <= 2**30
    assert shutil.disk_usage(ROOT).free >= 40*2**30


def no_learning(*args, **kwargs):
    raise AssertionError('T454 cannot optimize or backpropagate')


def configure():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.optim.Adam.step = no_learning
    torch.Tensor.backward = no_learning
    torch.autograd.grad = no_learning


def hidden(q, x):
    result, _ = q.core(q.encoder(q.normalized(x))[:, None])
    return result[:, 0]


def output(q, h):
    return 1.25*torch.tanh(q.head(h))


def statistics(delta):
    d = delta.detach().cpu().numpy()
    first = np.flatnonzero(np.max(np.abs(d), axis=1) > .01)
    return dict(rms=float(np.sqrt(np.mean(d*d))), max_abs=float(np.abs(d).max()),
        drive_mean=float(d[:, 0].mean()), steer_mean=float(d[:, 1].mean()),
        drive_rms=float(np.sqrt(np.mean(d[:, 0]**2))), steer_rms=float(np.sqrt(np.mean(d[:, 1]**2))),
        first20_drive_mean=float(d[:20, 0].mean()), first20_steer_mean=float(d[:20, 1].mean()),
        first_delta_gt_point01_s=None if len(first) == 0 else float(first[0]*.2),
        array_digest=digest(d))


def calculate(p):
    configure()
    assert all(not Path('/proc', str(pid)).exists() for pid in p['previous_scientific_pids'])
    models = [P.frozen(p, m['path']) for m in p['models']]
    result = dict(scope='Identical supplied real histories; block decompositions descriptive, not causal closed-loop tests',
        tasks=[], parameter_changes=[], pair_order=['parent_to_warmup', 'warmup_to_PPO0', 'PPO0_to_PPO1'],
        native=0, optimizers=0, newly_deployed_models=0)
    payload = {}
    with torch.no_grad():
        for before, after, stage in zip(models, models[1:], result['pair_order']):
            row = dict(stage=stage)
            for block, prefixes in [('representation', ('encoder.', 'core.')), ('head', ('head.',))]:
                a = torch.cat([v.reshape(-1) for n, v in before.named_parameters() if n.startswith(prefixes)])
                b = torch.cat([v.reshape(-1) for n, v in after.named_parameters() if n.startswith(prefixes)])
                row[block] = dict(parameters=a.numel(), delta_l2=float(torch.linalg.vector_norm(b-a)),
                    initial_l2=float(torch.linalg.vector_norm(a)), relative_l2=float(torch.linalg.vector_norm(b-a)/torch.linalg.vector_norm(a)))
            result['parameter_changes'].append(row)
        for t, ref in enumerate(p['histories']):
            assert sha(ref['path']) == ref['sha256']
            a = E.N.S.arrays(ref['path'])
            assert digest(a) == ref['array_digest']
            x = torch.tensor(a['states'])
            hs = [hidden(q, x) for q in models]
            locations = [output(q, h) for q, h in zip(models, hs)]
            for q, loc in zip(models, locations):
                assert torch.equal(q.sequence(x), loc)
            # The parent was the behavior model for all supplied histories.
            assert np.array_equal(locations[0].clamp(-1, 1).float().numpy(), a['actions'])
            stepped, h = [], None
            for i, row in enumerate(x):
                hb = np.zeros(64) if h is None else h[0, 0].numpy().copy()
                assert np.array_equal(hb, a['hidden_before'][i])
                command, loc, h, _ = models[0].step(row[None], h)
                assert np.array_equal(command[0].numpy(), a['actions'][i])
                stepped.append(loc[0])
            assert np.array_equal(h[0, 0].numpy(), a['last_hidden_after'])
            assert float((torch.stack(stepped)-locations[0]).abs().max()) < 1e-10
            table = dict(job_id=ref['job_id'], reference_role=ref['role'], steps=len(x),
                reference_success=ref['success'], comparisons=[])
            for k, loc in enumerate(locations):
                payload[f't{t}_model{k}_locations'] = loc.numpy()
                payload[f't{t}_model{k}_hidden'] = hs[k].numpy()
            for k, stage in enumerate(result['pair_order']):
                before, after = models[k], models[k+1]
                old, new = locations[k], locations[k+1]
                head_first = output(after, hs[k])
                representation_first = output(before, hs[k+1])
                total = new-old
                h_first = head_first-old
                r_after = new-head_first
                r_first = representation_first-old
                h_after = new-representation_first
                assert float((h_first+r_after-total).abs().max()) < 1e-14
                assert float((r_first+h_after-total).abs().max()) < 1e-14
                table['comparisons'].append(dict(stage=stage, total=statistics(total),
                    executed_action_delta=statistics(new.clamp(-1, 1).float().double()-old.clamp(-1, 1).float().double()),
                    head_first=statistics(h_first), representation_after_head=statistics(r_after),
                    representation_first=statistics(r_first), head_after_representation=statistics(h_after)))
                payload[f't{t}_stage{k}_head_first_location'] = head_first.numpy()
                payload[f't{t}_stage{k}_representation_first_location'] = representation_first.numpy()
            result['tasks'].append(table)
    train_ids = [r['job_id'] for record in p['prior_rounds'] for r in record['train_rows']]
    result['actual_PPO_training_task_counts'] = {name: train_ids.count(name) for name in sorted(set(train_ids))}
    result['task2_new_PPO_training_episodes'] = sum(name.startswith(('dev_task2_', 'hold_task2_')) for name in train_ids)
    result['actual_PPO_all_training_episodes'] = len(train_ids)
    result['compact_panel_evidence'] = p['panel_evidence']
    result['array_digest'] = digest(payload)
    result['array_hashes'] = {k: digest(v) for k, v in payload.items()}
    return result, payload


def prepare():
    assert not OUT.exists(), 'Never restart'
    prior = ROOT/'stage/runs/t453_early_correction_ppo'
    base = read(prior/'protocol.json')
    s = read(prior/'summary.json')
    assert read(prior/'audit.json')['state'] == 'PASS'
    ids = read(prior/'SEALED.json')['scientific_pids']
    assert all(not Path('/proc', str(pid)).exists() for pid in ids)
    assert not Path('/proc', str(read(prior/'driver_started.json')['pid'])).exists()
    keys = ['source_protocol', 'initial_model', 'student_start_model', 'student_start_sha256',
        'raw_sigma', 'phase', 'relay_stage', 'input_mean', 'input_scale', 'actor_seed', 'value_seed']
    p = {k: copy.deepcopy(base[k]) for k in keys}
    parent_rows = list(map(json.loads, (ROOT/'stage/runs/t452_midstate_recovery/cases.jsonl').read_text().splitlines()))
    histories = [next(r for r in base['own_rows'] if r['job_id'] == name) for name in ['dev_task2_13', 'wide117_80_dev6_RIGHT_13']]
    histories.append(next(r for r in parent_rows if r['role'] == 'FROZEN_STUDENT_CONTROL'))
    rows = list(map(json.loads, (prior/'cases.jsonl').read_text().splitlines()))
    panel = [r for r in rows if r['job_id'] in {r['job_id'] for r in histories} and r['role'] in ['WARMUP_PANEL', 'PPO_PANEL_0', 'PPO_PANEL_1']]
    p.update(id='T454_FROZEN_FUNCTION_AND_BLOCK_AUDIT', out=str(OUT),
        models=[base['research_initial']]+s['models'], histories=histories,
        prior_rounds=s['rounds'], previous_scientific_pids=ids,
        panel_evidence=[{k: r[k] for k in ['job_id', 'role', 'success', 'collision', 'timeout', 'physics', 'remaining_m']} for r in panel],
        max_bytes=4*2**20, main_seconds=600, audit_seconds=600, backup_seconds=600, all_seconds=2000,
        native=0, optimizers=0, Actor_updates=0, fresh16=None, reload78=None,
        six_gate=False, eight_started=False, parent_T453_remote='PENDING_6_OF16_BINARY_SEGMENTS_VERIFIED_ORIGINAL_RAW_LOCAL',
        mechanism='Four existing frozen checkpoints on three identical real parent histories, exact head/representation telescoping decomposition both orders. No causal navigation claim, no hybrid controller/export.',
        selection='Fixed models/histories/comparisons, no parameter search, no new policy candidate.',
        user_authorization='Nov10 user continue after completed T453 PPO regression; new finite zero-native zero-optimizer localization only.')
    files = set(base['inputs']) | {str(f) for f in HERE.glob('*.py')}
    files |= {str(DOC/'T454_PLAN.md'), str(DOC/'T454_PRIMARY_SOURCES.md'), str(DOC/'T453_FINDINGS.md'),
        str(prior/'summary.json'), str(prior/'audit.json'), str(prior/'cases.jsonl'), str(prior/'SEALED.json')}
    files |= {m['path'] for m in p['models']} | {r['path'] for r in histories}
    p['inputs'] = {f: sha(f) for f in sorted(files)}
    assert used()+p['max_bytes'] < 2**30
    OUT.mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    verify(p)
    save(DOC/'T454_BUDGET.json', dict(state='REGISTERED_NOT_LAUNCHED', own_max=p['max_bytes'],
        global_full=used()-used(OUT)+p['max_bytes'], native=0, optimizers=0,
        main_seconds=600, audit_seconds=600, all_seconds=2000, expected_science_minutes=[1, 3],
        hard_deadline_NOT_queued_duration=True, no_new_models_or_copies=True, upstream_T453_backup_pending=True))
    print('T454_REGISTERED', flush=True)


def preflight():
    p = read(OUT/'protocol.json')
    verify(p)
    assert not (OUT/'preflight.json').exists()
    configure()
    q = P.frozen(p, p['models'][0]['path'])
    a = E.N.S.arrays(p['histories'][0]['path'])
    x = torch.tensor(a['states'][:8])
    with torch.no_grad():
        assert torch.equal(q.sequence(x), output(q, hidden(q, x)))
    assert all(sha(m['path']) == m['sha256'] for m in p['models'])
    save(OUT/'preflight.json', dict(state='PASS', native=0, optimizers=0, existing_models_and_inputs_SHA=True,
        same_sequence_formula_exact=True, no_simulator_or_environment_constructed=True))
    print('T454_PREFLIGHT_PASS', flush=True)


def main():
    p = read(OUT/'protocol.json')
    verify(p)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'summary.json').exists()
    start = time.time()
    result, arrays = calculate(p)
    assert time.time()-start < p['main_seconds']
    path = OUT/'function_arrays.npz'
    E.N.write_npz(path, arrays, 2*2**20)
    save(OUT/'findings.json', result)
    save(OUT/'summary.json', dict(state='COMPLETE', pid=os.getpid(), epoch=time.time(), elapsed_s=time.time()-start,
        native=0, optimizers=0, models_created=0, findings_sha256=sha(OUT/'findings.json'),
        arrays_sha256=sha(path), array_digest=result['array_digest'], six_gate=False, eight_started=False))
    verify(p)
    print('T454_MAIN_COMPLETE', flush=True)


def audit():
    p = read(OUT/'protocol.json')
    verify(p)
    s = read(OUT/'summary.json')
    assert s['state'] == 'COMPLETE' and not Path('/proc', str(s['pid'])).exists()
    assert not (OUT/'audit.json').exists()
    result, arrays = calculate(p)
    assert result == read(OUT/'findings.json')
    saved = dict(np.load(OUT/'function_arrays.npz'))
    assert digest(saved) == digest(arrays) == s['array_digest']
    assert all(np.array_equal(saved[k], v) for k, v in arrays.items())
    save(OUT/'audit.json', dict(state='PASS_FROZEN_DESCRIPTOR_ONLY', pid=os.getpid(), native=0,
        optimizers=0, full_h0_history_and_both_block_orders_exact=True,
        every_saved_array_recomputed_exact=True, source_and_original_model_SHA_unchanged=True,
        causal_or_safe_navigation_proof=False, six_gate=False, eight_started=False))
    print('T454_AUDIT_PASS', flush=True)


def launch():
    p = read(OUT/'protocol.json')
    verify(p)
    assert read(OUT/'preflight.json')['state'] == 'PASS' and not (OUT/'launch.json').exists()
    receipt = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], text=True).strip()
    assert not subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()
    now = time.time()
    with (OUT/'driver_console.log').open('xb') as f:
        c = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=2', str(p['all_seconds']),
            'taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/'driver.py')],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    save(OUT/'launch.json', dict(pid=c.pid, epoch=now, all_stop=now+p['all_seconds'],
        protocol_sha256=sha(OUT/'protocol.json'), source_remote=receipt['remote_commit']))
    print('T454_ACTUALLY_LAUNCHED', c.pid, flush=True)


if __name__ == '__main__':
    try:
        globals()[sys.argv[1]]()
    except BaseException:
        if OUT.exists():
            save(OUT/f'error_{sys.argv[1]}_{os.getpid()}.json', dict(error=traceback.format_exc(), epoch=time.time(), native=0, optimizers=0))
        raise
