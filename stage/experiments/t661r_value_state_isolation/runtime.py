"""One audited Actor proposal and finite, durable deterministic evaluation."""
from pathlib import Path
import copy
import fcntl
import gzip
import json
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t661r_value_state_isolation'
DOC = ROOT/'stage/docs/day_20261120/t661r'
sys.path.insert(0, str(HERE))
import t661r_learning as Learn
import t661r_wait as Wait
sys.path.insert(0, str(ROOT/'stage/experiments/t660s_remaining_night_relay'))
import t660_runtime as Base
M, R, O, N = Learn.M, Base.R, Base.O, Base.N
read, save, sha, digest = M.read, M.save, M.sha, M.digest


def plan():
    p = read(HERE/'protocol.json')
    assert sha(HERE/'protocol.json') == sha(OUT/'protocol.json')
    return p


def verify(p, *unused):
    assert p['authorization_approved'] and time.time() < p['absolute_all_stop']
    assert sha(HERE/'protocol.json') == sha(OUT/'protocol.json')
    assert R.used(OUT) <= p['max_bytes']
    assert R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'] < p['global_bytes']
    assert R.D.B.shutil.disk_usage(ROOT).free >= 40*2**30
    Wait.resources()
    receipt = read(DOC/'AFFINITY_GUARD.json')
    assert receipt['state'] == 'ACTIVE_RESOURCE_ONLY'
    assert Path('/proc', str(receipt['pid'])).exists()
    for file, expected in p['inputs'].items():
        assert sha(file) == expected, file


def setup(p):
    Base.verify = verify
    Base.Wait = Wait
    Base.DOC = DOC
    Base.setup(p)
    # Runtime.request and Runtime.guard use these module globals.
    O.OUT = R.OUT = OUT
    R.verify = verify
    assert R.Runtime.guard.__globals__['verify'] is verify
    assert R.N.Pool.__init__.__globals__['worker'] is N.worker
    original_gpu_guard = R.gpu_guard
    is_audit = len(sys.argv) > 1 and sys.argv[1] == 'audit'
    gpu_deadline = min(time.time()+p['audit_seconds'], p['absolute_all_stop']-60) if is_audit else p['absolute_learning_stop']
    def gpu_guard():
        return wait_gpu(original_gpu_guard, gpu_deadline)
    R.gpu_guard = gpu_guard


def wait_gpu(probe, deadline, sleep=time.sleep, clock=time.time):
    while clock() < deadline:
        try:
            return probe()
        except AssertionError as error:
            if error.args != ('Other GPU compute exists; yield',):
                raise
            sleep(.25)
    raise TimeoutError('Stage GPU yield reached original bounded deadline')


def event(who):
    return Base.event(OUT, who, 0)


def learn(p, rt):
    lock = Base.gpu(p, OUT)
    objects = []
    try:
        obj, shadow = M.make(p, 'cuda'), M.make(p, 'cuda')
        objects = [obj, shadow]
        start = Learn.initial_state(p)
        Learn.load(obj, start)
        Learn.load(shadow, start)
        assert digest(M.cpu_state(obj)) == digest(M.cpu_state(shadow)) == digest(start)
        start_record = M.save_state(OUT/'start.pt.gz', start, p['state_bytes'])
        started = time.time()
        def guard():
            rt.guard()
            assert time.time() < min(started+p['learning_seconds'], p['absolute_learning_stop'])-5
            R.gpu_guard()
        data = Learn.groups(p, obj[0], guard)
        advantage = Learn.advantage_report(obj, data, p)
        assert advantage['failed_with_nonzero_advantage'] > 0
        references = M.references(p, obj[0], guard)
        first = M.fit(obj, references, data, p, guard, event('live'))
        second = M.fit(shadow, references, data, p, guard, event('shadow'))
        assert first == second
        assert digest(M.cpu_state(obj)) == digest(M.cpu_state(shadow))
        trained = M.save_state(OUT/'trained.pt.gz', M.cpu_state(obj), p['state_bytes'])
        proposal = M.export(obj, p, OUT/'proposal.pt.gz')
        # A rejected Actor restores Actor/Adam, while keeping learned V/Adam.
        rejected = copy.deepcopy(start)
        end = M.cpu_state(obj)
        for key in ['value', 'value_opt']:
            rejected[key] = copy.deepcopy(end[key])
        Learn.load(obj, rejected)
        for key in ['actor', 'actor_opt']:
            assert digest(M.cpu_state(obj)[key]) == digest(start[key])
        for key in ['value', 'value_opt']:
            assert digest(M.cpu_state(obj)[key]) == digest(end[key])
        reject_record = M.save_state(OUT/'rejected_Actor_kept_Value.pt.gz', M.cpu_state(obj), p['state_bytes'])
        record = dict(start=start_record, trained=trained, proposal=proposal, update=first,
            advantages=advantage, rejected_Actor_kept_Value=reject_record,
            historical_zero_baseline_control=Learn.records(p)[1]['proposal'],
            baseline_trained_only_on_previous_block=True)
        save(OUT/'LEARNING.json', record)
        return record
    finally:
        Base.free_gpu(lock, objects)


def chunks(jobs, size):
    for first in range(0, len(jobs), size):
        yield first//size, jobs[first:first+size]


def chunk_time_reserve(jobs, p):
    return 60+sum(j['case']['step_cap'] for j in jobs)/p['admission_physics_per_second']


def main():
    p = plan()
    verify(p)
    setup(p)
    R.E.L.install_stop_handlers()
    lock = (ROOT/'stage/runs/.stage_writer.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    save(OUT/'science_started.json', dict(pid=os.getpid(), epoch=time.time()))
    rt = O.Runtime(p)
    error = record = screen = full = None
    try:
        record = learn(p, rt)
        Base.pool(rt)
        values = rt.request(p['screen_jobs'], record['proposal'], 'screen36', False)
        by = {row['job_id']: row for arrays, row in values}
        safe_ids = {key for key, row in by.items() if Base.Prior.safe(row)}
        lost = sorted(set(p['original_screen_success_ids'])-safe_ids)
        screen = dict(lost_old_ids=lost, safe_ids=sorted(safe_ids),
            field_success=sum(Base.Prior.safe(by[j['id']]) for j in p['field_jobs']),
            historical_zero_baseline_lost=Learn.records(p)[1]['lost_screen_ids'],
            rows=list(by.values()))
        save(OUT/'SCREEN.json', screen)
        keys = [by[j['id']] for j in p['key_jobs']]
        rt.request(p['key_jobs'], record['proposal'], 'exact20', True, keys)
        acquired = safe_ids-set(p['before_success_ids'])
        if not lost and acquired:
            rest = [j for j in p['evaluation_jobs'] if j['id'] not in by]
            assert len(rest) == 371
            # One pass, no unconditional 407x2 profile. Durable 48-ID blocks.
            for index, jobs in chunks(rest, p['evaluation_chunk_size']):
                if time.time()+chunk_time_reserve(jobs, p) >= rt.deadline-10:
                    save(OUT/'NO_FULL407.json', dict(reason='NEXT_COMPLETE_CHUNK_TIME_ADMISSION_DENIED',
                        completed_ids=sorted(by), remaining_ids=[j['id'] for j in p['evaluation_jobs'] if j['id'] not in by]))
                    break
                values = rt.request(jobs, record['proposal'], f'rest_{index:02d}', False)
                by.update({row['job_id']: row for arrays, row in values})
                save(OUT/'EVALUATION_PROGRESS.json', dict(completed_ids=sorted(by),
                    complete=False, remaining_ids=[j['id'] for j in p['evaluation_jobs'] if j['id'] not in by],
                    epoch=time.time()))
            if len(by) == 407:
                ids = {key for key, row in by.items() if Base.Prior.safe(row)}
                full = dict(success=len(ids), success_ids=sorted(ids),
                    lost_prior=sorted(set(p['before_success_ids'])-ids),
                    gained_prior=sorted(ids-set(p['before_success_ids'])),
                    all387=set(p['required387_ids']) <= ids,
                    accepted=set(p['before_success_ids']) <= ids and bool(ids-set(p['before_success_ids'])))
                save(OUT/'JOINT407.json', full)
        elif lost:
            save(OUT/'NO_FULL407.json', dict(reason='OLD_SCREEN_LOSS', lost=lost))
        else:
            save(OUT/'NO_FULL407.json', dict(reason='NO_SCREEN_ACQUISITION'))
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, raw_preserved=True, no_restart=True))
    finally:
        Base.close(rt)
        rt.commands += 3
        save(OUT/'summary.json', dict(state='FAILED' if error else 'COMPLETE', pid=os.getpid(),
            error=error, physics=sum(rt.counts), commands=rt.commands, elapsed_s=time.time()-rt.start,
            learning=record, screen=screen, full407=full,
            retained=record['proposal'] if full and full['accepted'] else p['initial'],
            historical_control_reused_not_new_sample=True))
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    assert error is None, error


def audit():
    p = plan()
    verify(p)
    setup(p)
    summary = read(OUT/'summary.json')
    assert summary['state'] == 'COMPLETE' and not Path('/proc', str(summary['pid'])).exists()
    for group in read(OUT/'worker_generations.json'):
        assert all(not Path('/proc', str(pid)).exists() for pid in [group['parent'], *group['workers']])
    rows = [json.loads(x) for x in gzip.decompress((OUT/'cases.jsonl.gz').read_bytes()).decode().splitlines()]
    by = {row['path']: row for row in rows}
    assert len(rows) == len(by) and sum(row['physics'] for row in rows) == summary['physics'] <= p['max_physics']
    assert len(rows)+6 == summary['commands'] <= p['max_commands']
    for row in rows:
        assert sha(row['path']) == row['sha256']
        arrays = R.E.N.S.arrays(row['path'])
        if row['full']:
            N.inspect(p, arrays, row)
        else:
            assert all(digest(value) == row['array_hashes'][key] for key, value in arrays.items())
        assert row['loaded_model_sha256'] == summary['learning']['proposal']['sha256']
        if row.get('exact'):
            assert row['array_hashes'] == by[row['reference_path']]['array_hashes']
            assert row['integration_sha256'] == by[row['reference_path']]['integration_sha256']
    gl = Base.gpu(p, OUT)
    obj = M.make(p, 'cuda')
    try:
        started = time.time()
        def guard():
            assert time.time() < min(started+p['audit_seconds']-30, p['absolute_all_stop']-60)
            verify(p)
            R.gpu_guard()
        Learn.load(obj, Learn.initial_state(p))
        record = summary['learning']
        assert digest(M.cpu_state(obj)) == record['start']['state_hash']
        data = Learn.groups(p, obj[0], guard)
        assert Learn.advantage_report(obj, data, p) == record['advantages']
        references = M.references(p, obj[0], guard)
        update = M.fit(obj, references, data, p, guard, event('independent_history'))
        assert update == record['update']
        assert digest(M.cpu_state(obj)) == record['trained']['state_hash']
        assert obj[0].phash() == record['proposal']['policy_hash']
        if summary['full407']:
            fresh = [row for row in rows if not row.get('exact')]
            assert len(fresh) == len({r['job_id'] for r in fresh}) == 407
            ids = {row['job_id'] for row in fresh if Base.Prior.safe(row)}
            assert sorted(ids) == summary['full407']['success_ids']
            assert summary['full407']['accepted'] == (set(p['before_success_ids']) <= ids and bool(ids-set(p['before_success_ids'])))
        calls = [json.loads(x) for x in (OUT/'optimizer_calls.jsonl').read_text().splitlines()]
        for who in ['live', 'shadow', 'independent_history']:
            for kind, expected in [('actor', update['Actor_Adam']), ('value', update['Value_Adam'])]:
                assert [v[3:] for v in calls if v[:3] == [who, 0, kind]] == [[stage, k] for k in range(expected) for stage in ['BEFORE', 'AFTER']]
        save(OUT/'audit.json', dict(state='PASS', complete_history_density_and_state_chains=True,
            independent_Actor=update['Actor_Adam'], independent_Value=update['Value_Adam'], new_native=0))
    finally:
        Base.free_gpu(gl, [obj])


if __name__ == '__main__':
    {'main': main, 'audit': audit}[sys.argv[1]]()
