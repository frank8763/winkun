"""Fixed-model gradient diagnosis. No optimizer, native simulation or policy fit."""
from pathlib import Path
import ast
import collections
import gzip
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'stage/runs/t378_gradient_conflict'
DOC = ROOT / 'stage/docs/narrow_20261105'
sys.path.insert(0, str(ROOT / 'stage/experiments/t377_full_human_observation_transfer'))
import t377_core as S

np, torch = S.np, S.torch


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, cap=192 * 1024):
    blob = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    assert len(blob) <= cap, (path, len(blob), cap)
    Path(path).write_bytes(blob)


def sha(path):
    return S.K.C.sha(str(path))


def used(path):
    return S.K.S.used(path)


def absent(pid):
    assert not Path('/proc', str(pid)).exists(), ('Process still present', pid)


def source_closed():
    old = ROOT / 'stage/runs/t377_full_human_observation_transfer'
    assert read(old / 'summary.json')['state'] == 'COMPLETE'
    assert read(old / 'audit.json')['state'] == 'PASS'
    receipt = read(ROOT / 'stage/docs/long_distance_20261101/release_receipt_T377.json')
    assert receipt['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    for pid in [read(old / 'summary.json')['pid'], read(old / 'launch.json')['pid'],
                read(DOC / 'T377_DRIVER.json')['pid'], *read(old / 'workers.json')['workers']]:
        absent(pid)


def verify(p):
    assert (HERE / 'protocol.json').read_bytes() == (OUT / 'protocol.json').read_bytes()
    for name, value in p['inputs'].items():
        assert sha(name) == value, name
    original = read(p['source_protocol'])
    for name, value in original['input_hashes'].items():
        assert sha(name) == value, name
    assert used(ROOT / 'stage/runs') - used(OUT) + p['final_bytes'] <= p['global_bytes']
    assert used(OUT) <= p['final_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']
    return original


def forbid(*_args, **_kwargs):
    raise AssertionError('T378 forbids optimizer steps and native/worker entry')


def isolate():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.optim.Adam.step = forbid
    torch.optim.SGD.step = forbid
    S.P.C.B.setup = forbid
    S.mp.get_context = forbid


def actor(p, checkpoint=None):
    q = S.NightActor(p, 0.).eval().requires_grad_(False)
    q.adapter.requires_grad_(True)
    if checkpoint:
        q.load_learned(S.read_model(checkpoint))
    assert sum(v.numel() for v in q.trainable()) == 558
    return q


def direction(human, old, factual):
    protection = old + factual
    total = human + protection
    def cosine(a, b):
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return None if denominator == 0 else float(a @ b / denominator)
    norm2 = float(human @ human)
    assert norm2 > 0
    span = np.stack([old, factual], axis=1)
    coefficient, _, rank, singular = np.linalg.lstsq(span, human, rcond=1e-12)
    residual = human - span @ coefficient
    return dict(human_norm=float(np.linalg.norm(human)), old_norm=float(np.linalg.norm(old)),
                factual_norm=float(np.linalg.norm(factual)), protection_norm=float(np.linalg.norm(protection)),
                total_norm=float(np.linalg.norm(total)), human_old_cosine=cosine(human, old),
                human_factual_cosine=cosine(human, factual), human_protection_cosine=cosine(human, protection),
                human_total_cosine=cosine(human, total), protection_dot_human=float(protection @ human),
                scalar_GD_human_descent_ratio=float(human @ total / norm2),
                scalar_GD_human_first_derivative=float(-human @ total),
                protection_span_rank=int(rank), protection_span_singular_values=singular.tolist(),
                human_outside_protection_span_norm=float(np.linalg.norm(residual)),
                human_outside_protection_span_fraction=float(np.linalg.norm(residual) / np.linalg.norm(human)),
                negative_gradient_interference=bool(protection @ human < 0),
                scope='Fixed histories and parameters; scalar infinitesimal GD only, not Adam or closed-loop causality')


def human_errors(q, demos):
    result = []
    with torch.no_grad():
        for d in demos:
            prediction = q.sequence(d['x'], d['b'])[d['first']:]
            target = d['y'][d['first']:]
            squared = (prediction - target).square()
            error = squared.numpy()
            change = np.r_[0., np.abs(np.diff(target[:, 1].numpy()))]
            rows = []
            for name, mask in [('steer_change_ge_005', change >= .05),
                               ('other_labels', change < .05)]:
                rows.append(dict(group=name, labels=int(mask.sum()),
                                 drive_steer_RMSE=None if not mask.any() else np.sqrt(error[mask].mean(0)).tolist()))
            quarters = [dict(labels=len(v), drive_steer_RMSE=np.sqrt(v.mean(0)).tolist())
                        for v in np.array_split(error, 4)]
            result.append(dict(id=d['id'], human_labels=len(error), masked_prefix=d['first'],
                               drive_steer_RMSE=squared.mean(0).sqrt().tolist(),
                               turn_groups=rows, chronological_quarters=quarters))
    return result


def compute(p, guard):
    original = verify(p)
    q0 = actor(original)
    data = S.dataset(original, q0, guard)
    old, demos, factual = data
    schedule = S.schedule(original, old)
    assert len(schedule) == 300 and len(old) == 211 and len(factual) == 14
    old_frequency = collections.Counter(schedule)
    factual_frequency = collections.Counter(k % 14 for k in range(300))
    arrays = {}
    reports = []
    gradient_calls = 0
    for name, checkpoint in [('initial', p['initial_checkpoint']), ('final', p['final_checkpoint'])]:
        q = actor(original, checkpoint)
        before = S.digest(q.learned())
        frozen_before = S.fixed_hash(q)
        assert frozen_before == original['repair_parent']['hashes']['learned']
        vectors = {}
        values = {}
        entries = []
        old_sum = np.zeros(558, dtype=np.float64)
        factual_sum = np.zeros(558, dtype=np.float64)
        for group, refs in [('human', demos), ('old', old), ('factual', factual)]:
            for index, ref in enumerate(refs):
                guard()
                output = q.sequence(ref['x'], ref['b'])
                if group == 'human':
                    error = (output[ref['first']:] - ref['y'][ref['first']:]).square()
                    loss = .5 * error.mean()
                else:
                    error = (output - ref['y']).square()
                    loss = error.mean() if group == 'old' else error.max()
                gradient = torch.autograd.grad(loss, q.trainable())
                gradient_calls += 1
                vector = torch.cat([v.reshape(-1) for v in gradient]).detach().numpy().copy()
                assert vector.shape == (558,) and np.isfinite(vector).all()
                entry = dict(group=group, index=index, id=ref.get('id'),
                             task=None if ref.get('task') is None else int(ref['task']),
                             decisions=len(ref['x']), weighted_loss=float(loss.detach()),
                             gradient_norm=float(np.linalg.norm(vector)), gradient_sha256=S.digest(vector))
                if group == 'human':
                    vectors['human' + str(index)] = vector
                    values['human' + str(index)] = float(loss.detach())
                elif group == 'old':
                    entry['schedule_visits'] = old_frequency[index]
                    old_sum += vector * (old_frequency[index] / 300.)
                    values['old_schedule_average'] = values.get('old_schedule_average', 0.) + float(loss.detach()) * old_frequency[index] / 300.
                    if index == schedule[-1]:
                        vectors['old_last_schedule'] = vector
                else:
                    entry['schedule_visits'] = factual_frequency[index]
                    factual_sum += vector * (factual_frequency[index] / 300.)
                    values['factual_schedule_average'] = values.get('factual_schedule_average', 0.) + float(loss.detach()) * factual_frequency[index] / 300.
                    if index == 299 % 14:
                        vectors['factual_last_schedule'] = vector
                entries.append(entry)
        human = vectors['human0'] + vectors['human1']
        vectors.update(human=human, old_schedule_average=old_sum, factual_schedule_average=factual_sum)
        for key, value in vectors.items():
            arrays[name + '_' + key] = value
        errors = human_errors(q, demos)
        expected = read(p['source_summary'])['result'][name + '_fit_metrics']['demonstrations']
        assert [{k: row[k] for k in ['id', 'human_labels', 'masked_prefix', 'drive_steer_RMSE']}
                for row in errors] == expected
        assert S.digest(q.learned()) == before and S.fixed_hash(q) == frozen_before
        assert all(v.grad is None for v in q.parameters())
        reports.append(dict(model=name, policy_hash=q.phash(), learned_sha256=before,
                            input_model_bytes_unchanged=True, loss_values=values, rows=entries,
                            human_errors=errors,
                            fixed_model_average_over_original300schedule=direction(human, old_sum, factual_sum),
                            fixed_model_last_schedule_item=direction(human, vectors['old_last_schedule'], vectors['factual_last_schedule'])))
    assert gradient_calls == p['gradient_calls_per_pass'] == 454
    return dict(state='COMPLETE', method='Two fixed checkpoints; original full-history loss and original300schedule weights',
                models=reports, native_commands=0, physics_steps=0, optimizer_calls=0,
                gradient_calls=gradient_calls, model_parameters_modified=False,
                independent_demonstrations=2, valid_human_labels=572,
                original_six_stable=False, old275=None, six54=None, fresh16=None, reload78=None,
                entered7or8=False, all_workers_stopped=True), arrays


def prepare():
    assert not OUT.exists()
    source_closed()
    source = ROOT / 'stage/runs/t377_full_human_observation_transfer'
    inputs = [source / 'protocol.json', source / 'summary.json', source / 'audit.json',
              source / 'initial_fallback.pt.gz', source / 'candidate.pt.gz',
              DOC / 'T378_PLAN.md', DOC / 'RESEARCH_RESET_20261105.md', *HERE.glob('*.py')]
    before = used(ROOT / 'stage/runs')
    p = dict(id='T378', source_protocol=str(source / 'protocol.json'), source_summary=str(source / 'summary.json'),
             initial_checkpoint=str(source / 'initial_fallback.pt.gz'), final_checkpoint=str(source / 'candidate.pt.gz'),
             inputs={str(f): sha(f) for f in sorted(inputs)}, native_commands=0, physics_steps=0,
             optimizer_calls=0, gradient_calls_per_pass=454, audit_gradient_calls=454,
             max_gradient_calls_total=908, synthetic_preflight_gradients=1,
             final_bytes=640 * 1024, global_bytes=1024**3, min_free_bytes=40 * 1024**3,
             main_seconds=600, audit_seconds=600, all_work_seconds=1800,
             zero_native_or_optimizer_preflight=True, unused_modes_not_launched=True)
    assert before + p['final_bytes'] <= p['global_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']
    OUT.mkdir()
    write(HERE / 'protocol.json', p, 16 * 1024)
    write(OUT / 'protocol.json', p, 16 * 1024)
    write(DOC / 'T378_BUDGET.json', dict(state='FULL_RESERVATION_PASS', global_before=before,
          reserved_bytes=p['final_bytes'], worst_payload_KiB=556, full_reservation=before+p['final_bytes'],
          global_limit=p['global_bytes'], native=0, optimizer=0, gradient_calls_total=908))
    print('REGISTERED_T378_ZERO_NATIVE_ZERO_OPTIMIZER')


def preflight():
    p = read(HERE / 'protocol.json')
    verify(p)
    assert not (OUT / 'preflight.json').exists()
    for f in HERE.glob('*.py'):
        ast.parse(f.read_text())
    isolate()
    x = torch.tensor([1., -2.], dtype=torch.float64, requires_grad=True)
    gradient = torch.autograd.grad(x.square().sum(), x)[0]
    assert torch.equal(gradient, 2*x)
    h = np.array([1., 0., 0.])
    same = direction(h, h*.5, np.zeros(3))
    conflict = direction(h, -h*2., np.zeros(3))
    zero = direction(h, np.zeros(3), np.zeros(3))
    assert same['scalar_GD_human_descent_ratio'] == 1.5
    assert conflict['scalar_GD_human_descent_ratio'] == -1. and conflict['negative_gradient_interference']
    assert zero['human_protection_cosine'] is None and zero['human_outside_protection_span_fraction'] == 1.
    write(OUT / 'preflight.json', dict(state='PASS', native=0, optimizer=0,
          synthetic_gradients=1, singular_zero_and_conflict_algebra=True, forbidden_entry_points_installed=True), 8*1024)
    print('PREFLIGHT_PASS_ZERO_NATIVE_ZERO_OPTIMIZER')


def main():
    p = read(HERE / 'protocol.json')
    verify(p)
    assert read(OUT / 'preflight.json')['state'] == 'PASS' and not (OUT / 'summary.json').exists()
    isolate()
    start = time.time()
    launch = read(OUT / 'launch.json')
    deadline = min(launch['main_deadline_epoch'], start + p['main_seconds'])
    write(OUT / 'workers.json', dict(parent=os.getpid(), workers=[]), 4*1024)
    last = 0.
    def guard():
        nonlocal last
        assert time.time() < deadline-3
        if time.time()-last > 3:
            last = time.time()
            verify_budget(p)
            write(OUT / 'status.json', dict(state='FIXED_MODEL_GRADIENTS', pid=os.getpid(), updated_epoch=last,
                  main_deadline_epoch=deadline, native=0, optimizer=0), 8*1024)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T378 main deadline')))
    signal.alarm(max(1, int(deadline-time.time())-2))
    try:
        report, vectors = compute(p, guard)
        write(OUT / 'results.json', report)
        buf = io.BytesIO()
        np.savez_compressed(buf, **vectors)
        assert len(buf.getvalue()) <= 96*1024
        (OUT / 'gradients.npz').write_bytes(buf.getvalue())
        summary = dict(state='COMPLETE', pid=os.getpid(), elapsed_s=time.time()-start,
                       native_commands=0, physics_steps=0, optimizer_calls=0, gradient_calls=454,
                       result_sha256=sha(OUT / 'results.json'), vector_hashes={k:S.digest(v) for k,v in vectors.items()},
                       vector_file_sha256=sha(OUT / 'gradients.npz'), all_workers_stopped=True,
                       no_policy_training=True, stable_original_six=False, entered7or8=False)
    except BaseException:
        write(OUT / 'failure.json', dict(error=traceback.format_exc(), native=0, optimizer=0), 32*1024)
        summary = dict(state='FAILED', pid=os.getpid(), elapsed_s=time.time()-start,
                       native_commands=0, physics_steps=0, optimizer_calls=0, all_workers_stopped=True)
        raise
    finally:
        signal.alarm(0)
        write(OUT / 'summary.json', summary, 12*1024)
        write(OUT / 'status.json', summary, 8*1024)
        verify_budget(p)
    print(json.dumps(summary))


def verify_budget(p):
    assert used(OUT) <= p['final_bytes']
    assert used(ROOT / 'stage/runs') <= p['global_bytes']
    assert shutil.disk_usage(ROOT).free >= p['min_free_bytes']


def audit():
    p = read(HERE / 'protocol.json')
    verify(p)
    assert not (OUT / 'audit.json').exists()
    summary = read(OUT / 'summary.json')
    assert summary['state'] == 'COMPLETE'
    absent(summary['pid'])
    absent(read(OUT / 'launch.json')['pid'])
    isolate()
    end = min(time.time()+p['audit_seconds'], read(OUT/'launch.json')['all_work_deadline_epoch']-20)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('T378 audit deadline')))
    signal.alarm(max(1, int(end-time.time())-2))
    def guard():
        assert time.time() < end-3
        verify_budget(p)
    report, vectors = compute(p, guard)
    assert report == read(OUT / 'results.json')
    assert sha(OUT / 'results.json') == summary['result_sha256']
    assert sha(OUT / 'gradients.npz') == summary['vector_file_sha256']
    with np.load(OUT / 'gradients.npz') as stored:
        assert set(stored.files) == set(vectors)
        assert all(np.array_equal(stored[k], v) and S.digest(v)==summary['vector_hashes'][k] for k,v in vectors.items())
    result = dict(state='PASS', native=0, optimizer=0, independent_gradient_recomputations=454,
                  all_454_row_gradient_hashes_and_aggregate_arrays_exact=True,
                  all_input_model_bytes_and_reference_metrics_exact=True, all_processes_absent=True,
                  results_sha256=summary['result_sha256'], gradients_sha256=summary['vector_file_sha256'])
    write(OUT / 'audit.json', result, 16*1024)
    write(DOC / 'T378_AUDIT.json', result, 16*1024)
    signal.alarm(0)
    print(json.dumps(result))


def launch():
    p = read(HERE/'protocol.json')
    verify(p)
    source_closed()
    assert not (OUT/'launch.json').exists() and read(OUT/'preflight.json')['state']=='PASS'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
    receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
    epoch=time.time()
    with (OUT/'main.log').open('xb') as log:
        process=subprocess.Popen(['systemd-inhibit','--what=sleep','--mode=block','--why=Stage zero-native gradient diagnosis',
              'timeout','--signal=TERM','--kill-after=1s','600s','taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),'main'],
              cwd=ROOT,env=dict(os.environ,STAGE_PROCESS_CPU='0'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(OUT/'launch.json',dict(pid=process.pid,launch_epoch=epoch,main_deadline_epoch=epoch+600,
          all_work_deadline_epoch=epoch+1800,source_remote_commit=receipt['remote_commit'],root_tree=receipt['root_tree'],
          protocol_sha256=sha(HERE/'protocol.json')),8*1024)
    print(json.dumps(dict(state='LAUNCHED_ZERO_NATIVE_DIAGNOSIS',pid=process.pid)))


if __name__=='__main__':
    assert sys.argv[1] in ['prepare','preflight','launch','main','audit']
    globals()[sys.argv[1]]()
