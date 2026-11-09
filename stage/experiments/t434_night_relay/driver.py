"""Bounded offline Stage relay, with explicit predecessor handoff.

Import is inert. The predecessor stays immutable and owns every scientific
write until its entire native/driver/timeout/inhibitor chain has exited.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os, sys, time, json, signal, subprocess, traceback, shutil, fcntl
import t434_core as C

ROOT, HERE, OUT, DOC = C.ROOT, C.HERE, C.OUT, C.DOC
read, save, sha = C.read, C.save, C.sha
LEARNING_STAGES = ['fresh_rl', 'iterative_recovery', 'representation_rl']
STAGES = [*LEARNING_STAGES, 'gate', 'eight']
PREVIOUS_OUT = ROOT / 'stage/runs/t432_offline_night'
PREVIOUS_DOC = ROOT / 'stage/docs/overnight_20261108'
ACTIVE = ACTIVE_PGID = ACTIVE_STARTED = None


def used(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


def process_info(pid):
    try:
        bits = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), state=bits[0], parent=int(bits[1]),
                    pgrp=int(bits[2]), start_ticks=int(bits[19]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def group_members(pgid):
    values = []
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            row = process_info(int(p.name))
            if row is not None and row['pgrp'] == pgid and row['state'] != 'Z':
                values.append(row)
    return values


def stop_active():
    global ACTIVE, ACTIVE_PGID, ACTIVE_STARTED
    if ACTIVE_PGID is None:
        return []
    group = ACTIVE_PGID
    for sig, seconds in [(signal.SIGTERM, 3.), (signal.SIGKILL, 2.)]:
        rows = group_members(group)
        if not rows:
            break
        if ACTIVE_STARTED is not None:
            assert all(r['start_ticks'] >= ACTIVE_STARTED for r in rows), 'Recycled child group'
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            pass
        end = time.monotonic() + seconds
        while group_members(group) and time.monotonic() < end:
            time.sleep(.05)
    if ACTIVE is not None:
        try:
            ACTIVE.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass
    remaining = group_members(group)
    ACTIVE = ACTIVE_PGID = ACTIVE_STARTED = None
    return remaining


def native_pids(folder):
    folder = Path(folder)
    ids, started = set(), False
    for name in ['main_started.json', 'workers.json', 'summary.json']:
        file = folder / name
        if not file.exists():
            continue
        row = read(file)
        started |= name == 'main_started.json'
        ids.update(row[k] for k in ['pid', 'parent', 'main_pid'] if type(row.get(k)) is int)
        ids.update(row.get('workers', []))
    live = [process_info(pid) for pid in sorted(ids) if process_info(pid) is not None]
    return dict(started=started, pids=sorted(ids), live=live,
                summary_exists=(folder / 'summary.json').exists(),
                all_native_processes_stopped=not live)


def check_stopped(folder, require_complete=False):
    result = native_pids(folder)
    assert result['started'] and result['pids'], ('No native launch evidence', str(folder))
    assert result['all_native_processes_stopped'], ('Native tree or recycled PID remains', result)
    if require_complete:
        summary = read(Path(folder) / 'summary.json')
        assert summary['state'] == 'COMPLETE'
        assert summary['all_workers_stopped'] and summary['all_sessions_closed']
    return result


def member_manifest(folder):
    folder = Path(folder)
    values = [dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=sha(f))
              for f in sorted(folder.rglob('*'))
              if f.is_file() and f.name != 'local_manifest.json']
    save(folder / 'local_manifest.json', dict(state='LOCAL_ALL_BYTES_SHA_VERIFIED_REMOTE_PENDING',
         files=values, epoch=time.time(), network_required=False, remote_verified=False))
    return check_manifest(folder)


def check_manifest(folder):
    file = Path(folder) / 'local_manifest.json'
    for row in read(file)['files']:
        path = ROOT / row['path']
        assert path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], row['path']
    return sha(file)


def hard_deadlines(p, launch):
    zone = ZoneInfo('Asia/Shanghai')
    fixed = {name: datetime(2026, 11, 9, hour, minute, tzinfo=zone).timestamp()
             for name, hour, minute in [('learning', 7, 0), ('native', 7, 45), ('all', 8, 30)]}
    return dict(learning=min(p['deadline_learning'], fixed['learning']),
                native=min(p['deadline_native'], fixed['native']),
                all=min(p['deadline_all'], fixed['all'], launch['deadline']))


def root_limits(p):
    values = p['root_limits']
    result = dict(physics=values.get('max_physics', 40000000),
                  commands=values.get('max_commands', 2000),
                  bytes=values['max_bytes'],
                  main_wall_s=p.get('maximum_main_wall_seconds', 18000))
    assert 0 < result['physics'] <= 40000000 and 0 < result['commands'] <= 2000
    assert 0 < result['bytes'] <= 2**30 and 0 < result['main_wall_s'] <= 18000
    return result


def call(file, args, seconds, deadline, log, status, heartbeat=True):
    global ACTIVE, ACTIVE_PGID, ACTIVE_STARTED
    assert ACTIVE is None and ACTIVE_PGID is None
    seconds = min(float(seconds), deadline - time.time() - 5)
    assert seconds > 0, ('No child time', file, args)
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('xb') as stream:
        ACTIVE = subprocess.Popen(['taskset', '-c', '0', str(ROOT / 'run_stage.sh'),
            str(HERE / file), *map(str, args)], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        ACTIVE_PGID = ACTIVE.pid
        row = process_info(ACTIVE.pid)
        ACTIVE_STARTED = None if row is None else row['start_ticks']
        end = time.time() + seconds
        print(json.dumps(dict(event='CHILD_STARTED', file=file, args=list(map(str, args)),
              pid=ACTIVE.pid, pgrp=ACTIVE_PGID, deadline=end)), flush=True)
        try:
            while ACTIVE.poll() is None:
                assert time.time() < end, ('Relative or absolute child cutoff', file, args)
                if heartbeat:
                    save(DOC / 'DRIVER_STATUS.json', dict(state=status, epoch=time.time(),
                        driver_pid=os.getpid(), child_pid=ACTIVE.pid, child_pgrp=ACTIVE_PGID,
                        child_file=file, args=list(map(str, args)), child_deadline=end,
                        offline_science=True))
                time.sleep(min(5., max(.05, end - time.time())))
            code = ACTIVE.returncode
            remaining = stop_active()
            assert not remaining, ('Child descendants remain', file, remaining)
            assert code == 0, (file, args, code)
            print(json.dumps(dict(event='CHILD_FINISHED', file=file,
                                 args=list(map(str, args)), exit_code=code)), flush=True)
        except BaseException:
            remaining = stop_active()
            if remaining:
                save(DOC / 'UNCLOSED_CHILD_GROUP.json', dict(epoch=time.time(),
                     file=file, members=remaining))
            raise


def predecessor_processes(p):
    """Captured start ticks make PID reuse distinguishable from a live old job."""
    rows = p.get('predecessor_processes')
    if rows is not None:
        assert rows and all(type(r['pid']) is int and type(r['start_ticks']) is int for r in rows)
        return rows
    launch = read(PREVIOUS_OUT / 'launch.json')
    started = read(PREVIOUS_OUT / 'driver_started.json')
    ids = {launch['pid'], started['pid'], started['parent']}
    return [dict(pid=pid, start_ticks=None) for pid in sorted(ids)]


def wait_predecessor(p, cut):
    launch_file = PREVIOUS_OUT / 'launch.json'
    wanted = p.get('predecessor_launch_sha256')
    if wanted is not None:
        assert sha(launch_file) == wanted, 'Predecessor launch identity changed'
    identities = predecessor_processes(p)
    while True:
        live = []
        for identity in identities:
            actual = process_info(identity['pid'])
            if actual is not None and actual['state'] != 'Z':
                ticks = identity.get('start_ticks')
                if ticks is None or actual['start_ticks'] == ticks:
                    live.append(actual)
        if not live:
            break
        assert time.time() + 900 < cut['all'], 'Predecessor consumed finalization reserve'
        save(DOC / 'DRIVER_STATUS.json', dict(state='WAITING_FOR_T432_ALL_WRITERS_TO_EXIT',
             epoch=time.time(), driver_pid=os.getpid(), predecessor_live=live,
             no_native_or_optimizer_started=True, offline_science=True))
        time.sleep(5.)
    seal = read(PREVIOUS_OUT / 'SEALED.json')
    report = read(PREVIOUS_DOC / 'FINAL_REPORT.json')
    assert seal['all_scientific_processes_stopped'] and seal['no_future_scientific_writes']
    assert report['all_scientific_processes_stopped']
    assert report['state'] == 'COMPLETE_AUDITED_LOCAL', 'Predecessor contract failure stops relay'
    check_manifest(PREVIOUS_OUT)
    prior = read(PREVIOUS_OUT / 'protocol.json')
    computed = dict(physics=0, commands=prior.get('preflight_zero_physics_commands', 0), main_wall_s=0.)
    evidence = []
    for alias in ['data', 'b', 'c', 'gate', 'eight']:
        folder = PREVIOUS_OUT / alias
        if not (folder / 'main_started.json').exists():
            continue
        native = check_stopped(folder, True)
        summary, audit = read(folder / 'summary.json'), read(folder / 'audit.json')
        assert audit['state'] == 'PASS', ('Predecessor audit not PASS', alias)
        check_manifest(folder)
        rows = [json.loads(line) for line in (folder / 'commands.jsonl').read_text().splitlines()]
        assert len(rows) == summary['commands']
        assert sum(row['physics'] for row in rows) == summary['physics']
        computed['physics'] += summary['physics']
        computed['commands'] += summary['commands']
        computed['main_wall_s'] += summary['elapsed_s']
        evidence.append(dict(stage=alias, summary_sha256=sha(folder / 'summary.json'),
            audit_sha256=sha(folder / 'audit.json'), native=native,
            local_manifest_sha256=sha(folder / 'local_manifest.json')))
    for key in computed:
        assert computed[key] == report['actual'][key], ('Predecessor ledger mismatch', key)
    proof = dict(state='PREDECESSOR_NATIVE_DRIVER_AND_FINALIZERS_STOPPED',
        actual=computed, processes=identities, evidence=evidence,
        launch_sha256=sha(launch_file), seal_sha256=sha(PREVIOUS_OUT / 'SEALED.json'),
        report_sha256=sha(PREVIOUS_DOC / 'FINAL_REPORT.json'),
        six_gate=bool(report['six_gate']), eight_started=bool(report['eight_started']))
    save(OUT / 'predecessor_handoff.json', proof)
    return proof


def main():
    p = read(OUT / 'protocol.json')
    C.verify(p)
    end = time.monotonic() + 10.
    while not (OUT / 'launch.json').exists():
        assert time.monotonic() < end, 'Missing actual launch receipt'
        time.sleep(.05)
    launch = read(OUT / 'launch.json')
    cut, limit = hard_deadlines(p, launch), root_limits(p)
    assert not (OUT / 'driver_started.json').exists()
    assert time.time() + 60 < cut['all']
    started, state, reason, error = time.time(), 'FAILED', 'UNFINISHED', None
    records, predecessor = [], None
    six_gate, eight_started = False, False
    remote, source = 'REMOTE_PENDING', 'SOURCE_PENDING'
    final_reserve = max(900., p.get('backup_seconds', 600) + p.get('source_seconds', 180) + 120.)
    save(OUT / 'driver_started.json', dict(pid=os.getpid(), parent=os.getppid(), epoch=started,
        deadline=cut['all'], learning_deadline=cut['learning'], native_deadline=cut['native'],
        offline_science=True))
    lock = (OUT / '.driver.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def interrupt(signum, frame):
        raise TimeoutError('T434 registered hard stop or interruption ' + str(signum))
    for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGALRM]:
        signal.signal(sig, interrupt)
    signal.alarm(max(1, int(cut['all'] - time.time() - 30)))
    def relay_actual():
        known = [row['summary'] for row in records if row.get('summary') is not None]
        return dict(physics=sum(row['physics'] for row in known),
            commands=p.get('preflight_zero_physics_commands', 0) + sum(row['commands'] for row in known),
            main_wall_s=sum(row['elapsed_s'] for row in known))
    def totals():
        own = relay_actual()
        old = predecessor['actual'] if predecessor is not None else dict(physics=0, commands=0, main_wall_s=0.)
        return {key: own[key] + old[key] for key in own}
    def ledger(stage):
        actual = totals()
        assert actual['physics'] <= limit['physics'] and actual['commands'] <= limit['commands']
        assert actual['main_wall_s'] <= limit['main_wall_s']
        assert used(OUT) <= limit['bytes'] and used(ROOT / 'stage/runs') <= 2**30
        assert shutil.disk_usage(ROOT).free >= 40 * 2**30
        save(DOC / 'DAY_LEDGER.json', dict(state=stage, epoch=time.time(), driver_pid=os.getpid(),
            deadline=cut, actual=actual, combined_night_counts_verified=predecessor is not None, relay_actual=relay_actual(), predecessor=predecessor,
            records=records, six_gate=six_gate, eight_started=eight_started,
            source=source, remote=remote, all_science_offline=True))
        return actual
    try:
        predecessor = wait_predecessor(p, cut)
        ledger('PREDECESSOR_HANDOFF_VERIFIED')
        if predecessor['six_gate']:
            six_gate = True
            reason = 'T432_ALREADY_PASSED_FULL_SIX_GATE_NO_DUPLICATE_FRESH_OR_EIGHT'
        else:
            for name in STAGES:
                folder = OUT / name
                budget = p['stage_budgets'][name]
                actual = ledger('ADMISSION_' + name)
                complete = [row for row in records if row['state'] == 'COMPLETE_AUDITED_LOCAL']
                why = []
                full = [row for row in complete if row['name'] in LEARNING_STAGES
                        and row['summary']['result'].get('development_all_safe') is True
                        and row['summary']['result'].get('accepted') is True]
                if name in LEARNING_STAGES and full:
                    why.append('PREVIOUS_LEARNING_STAGE_ALREADY_PASSED_FULL329')
                if name == 'gate' and not full:
                    why.append('NO_ACCEPTED_FULL329_MODEL')
                if name == 'eight' and not six_gate:
                    why.append('FULL_SIX_GATE_NOT_PASSED')
                reserve = dict(physics=0, commands=0, main_wall_s=0., bytes=0)
                if name in LEARNING_STAGES and p.get('reserve_gate_for_learning', True):
                    gate = p['stage_budgets']['gate']
                    reserve = dict(physics=gate['max_physics'], commands=gate['max_commands'],
                                   main_wall_s=gate['main_seconds'], bytes=gate['max_bytes'])
                    if p.get('reserve_eight_for_learning', False):
                        eight = p['stage_budgets']['eight']
                        reserve['physics'] += eight['max_physics']
                        reserve['commands'] += eight['max_commands']
                        reserve['main_wall_s'] += eight['main_seconds']
                        reserve['bytes'] += eight['max_bytes']
                if actual['physics'] + budget['max_physics'] + reserve['physics'] > limit['physics']:
                    why.append('NO_FULL_COMBINED_NIGHT_PHYSICS_AND_GATE_RESERVE')
                if actual['commands'] + budget['max_commands'] + reserve['commands'] > limit['commands']:
                    why.append('NO_FULL_COMBINED_NIGHT_COMMAND_AND_GATE_RESERVE')
                if actual['main_wall_s'] + budget['main_seconds'] + reserve['main_wall_s'] > limit['main_wall_s']:
                    why.append('NO_FULL_COMBINED_NIGHT_MAIN_WALL_AND_GATE_RESERVE')
                complete_bytes = budget['max_bytes'] + reserve['bytes']
                if (used(OUT) + complete_bytes > limit['bytes']
                    or used(ROOT / 'stage/runs') + complete_bytes > 2**30
                    or shutil.disk_usage(ROOT).free < 40 * 2**30):
                    why.append('NO_FULL_OWN_GLOBAL_AND_GATE_CAPACITY_RESERVE')
                now = time.time()
                if now + budget['main_seconds'] + 5 >= cut['native']:
                    why.append('NO_FULL_NATIVE_TIME_RESERVE')
                if name in LEARNING_STAGES:
                    validation = p.get('validation_reserve_seconds', 1200)
                    if now + budget['main_seconds'] - validation + 5 >= cut['learning']:
                        why.append('NO_FULL_LEARNING_TIME_RESERVE')
                needed = budget['main_seconds'] + budget['audit_seconds'] + 120 + final_reserve
                if now + needed >= cut['all']:
                    why.append('NO_FULL_AUDIT_MANIFEST_AND_FINALIZATION_TIME_RESERVE')
                if why:
                    records.append(dict(name=name, folder=str(folder), state='NOT_STARTED',
                                        reasons=why, summary=None))
                    ledger('SKIPPED_' + name)
                    continue
                assert not (folder / 'main_started.json').exists() and not (folder / 'summary.json').exists(), ('Duplicate stage', name)
                record = dict(name=name, folder=str(folder), state='RUNNING', start_epoch=time.time(), summary=None)
                records.append(record)
                if name == 'eight':
                    eight_started = True
                ledger('NATIVE_' + name)
                call('run.py', [name], budget['main_seconds'], cut['native'],
                     DOC / (name + '_MAIN.log'), 'NATIVE_' + name)
                record['native_stop_evidence'] = check_stopped(folder, True)
                summary = read(folder / 'summary.json')
                record['summary'] = summary
                assert summary['physics'] <= budget['max_physics'] and summary['commands'] <= budget['max_commands']
                assert summary['elapsed_s'] <= budget['main_seconds']
                ledger('AUDIT_' + name)
                call('audit.py', [name], budget['audit_seconds'], cut['all'] - final_reserve,
                     DOC / (name + '_AUDIT.log'), 'AUDIT_' + name)
                assert read(folder / 'audit.json')['state'] == 'PASS'
                record['native_stop_evidence'] = check_stopped(folder, True)
                record['manifest_sha256'] = check_manifest(folder)
                record['state'] = 'COMPLETE_AUDITED_LOCAL'
                if name == 'gate':
                    six_gate = summary['result'].get('six_gate') is True
                ledger('COMPLETE_' + name)
            reason = 'ALL_PREREGISTERED_RELAY_MECHANISMS_COMPLETED_OR_BOUNDED_SKIPPED'
        state = 'COMPLETE_AUDITED_LOCAL'
    except BaseException:
        error = traceback.format_exc()
        reason = 'CONTRACT_FAILURE_INTERRUPTION_OR_HARD_DEADLINE'
        if stop_active():
            state = 'FAILED_UNCLOSED_CHILD_GROUP'
        save(DOC / 'DRIVER_ERROR.json', dict(error=error, epoch=time.time(), no_science_restart=True))
        for record in records:
            if record['state'] != 'RUNNING':
                continue
            file = Path(record['folder']) / 'summary.json'
            record['summary'] = read(file) if file.exists() else None
            record['state'] = 'FAILED_RAW_PRESERVED' if file.exists() else 'INTERRUPTED_NO_SUMMARY_RAW_PRESERVED'
    finally:
        remaining = stop_active()
        native_trees = []
        closed = not remaining
        for name in STAGES:
            folder = OUT / name
            if not folder.exists():
                continue
            evidence = native_pids(folder)
            native_trees.append(dict(name=name, **evidence))
            if evidence['started'] and not evidence['all_native_processes_stopped']:
                closed = False
        if not closed:
            state = 'FAILED_UNCLOSED_NATIVE_TREE'
        save(OUT / 'SEALED.json', dict(state=state, reason=reason, epoch=time.time(),
            all_scientific_processes_stopped=closed, native_trees=native_trees,
            error=error, six_gate=six_gate, eight_started=eight_started,
            no_future_scientific_writes=True,
            missing_summary_is_interruption_not_completed_failure=True))
        member_manifest(OUT)
        actual = ledger('SCIENTIFIC_RELAY_SEALED')
        if closed and predecessor is not None and time.time() + 60 < cut['all']:
            try:
                receipt = ROOT / 'stage/docs/long_distance_20261101/release_receipt_T434.json'
                if not receipt.exists():
                    call('transport.py', ['T434'], p.get('backup_seconds', 600), cut['all'] - 240,
                         DOC / 'BACKUP.log', 'OPTIONAL_REMOTE_BACKUP')
                assert read(receipt)['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
                remote = read(receipt)['state']
            except BaseException:
                save(DOC / 'REMOTE_PENDING.json', dict(error=traceback.format_exc(), epoch=time.time(),
                    all_science_already_finished=True))
        report = dict(state=state, reason=reason, finished_epoch=time.time(), elapsed_s=time.time() - started,
            deadline=cut, records=records, actual=actual if predecessor is not None else None, relay_actual=relay_actual(), predecessor=predecessor,
            all_scientific_processes_stopped=closed, native_trees=native_trees, six_gate=six_gate,
            seven_skipped_by_explicit_user_request=True, eight_started=eight_started,
            remote=remote, source='CHECK_LATEST_SOURCE_SYNC_RECEIPT_AFTER_FINAL_COMMIT', error=error,
            safe_T255_not_promoted=True, narrow_paused=True,
            no_missing_summary_or_unexecuted_group_fabricated=True)
        save(DOC / 'FINAL_REPORT.json', report)
        (DOC / 'FINAL_REPORT.md').write_text('# Stage 11月8日夜间接力真实结果\n\n' +
            f'状态：{state}。停止理由：{reason}。\n\n' +
            (f'T432加接力合计物理{actual["physics"]}、命令{actual["commands"]}、主管线{actual["main_wall_s"]:.3f}秒。' if predecessor is not None else '前驱交棒证据未完整核验，合计夜间计数缺测；本接力未启动科学阶段。') +
            f'六米完整门槛={six_gate}；七米按用户要求跳过；本接力八米实际启动={eight_started}。\n\n' +
            '逐阶段已执行、未执行及失败以原始summary/audit为准，未执行null不是失败率。' +
            '研究候选不替换safeT255，窄场景暂停，Stage与Phase隔离。\n\n' +
            f'全部科学退出={closed}；远端={remote}；北京时间11月9日08:30全部硬停。\n')
        save(DOC / 'DRIVER_STATUS.json', dict(state=state, epoch=time.time(), driver_pid=os.getpid(),
            all_scientific_processes_stopped=closed, six_gate=six_gate, eight_started=eight_started, remote=remote))
        try:
            assert closed and predecessor is not None and time.time() + 90 < cut['all']
            status_file = ROOT / 'stage/docs/RESEARCH_STATUS.md'
            status_updated = False
            if launch.get('research_status_sha256') is not None and sha(status_file) == launch['research_status_sha256']:
                status_file.write_text('2026-11-08 T434离线接力结束，state=' + state + '；six_gate=' + str(six_gate) +
                    '，按用户授权跳7m，eight_started=' + str(eight_started) + '。全部科学退出=' + str(closed) +
                    '；remote=' + remote + '。T432加接力合计预算与真实缺测见overnight_20261108_relay/FINAL_REPORT.json；' +
                    '不重启结束目录、不晋升safeT255、窄暂停、Stage/Phase隔离。\n\n' + status_file.read_text())
                status_updated = True
            files = [DOC / 'FINAL_REPORT.json', DOC / 'FINAL_REPORT.md', DOC / 'DAY_LEDGER.json', DOC / 'DRIVER_STATUS.json']
            if status_updated:
                files.append(status_file)
            for name in ['release_receipt_T434.json', 'source_manifest_T434.json', 'T434_small_parts_progress.json']:
                path = ROOT / 'stage/docs/long_distance_20261101' / name
                if path.exists():
                    files.append(path)
            subprocess.run(['git', 'add', '--', *map(str, files)], cwd=ROOT, check=True, timeout=20)
            # Do not accidentally commit another process/user's already-staged files.
            staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT, text=True).splitlines()
            allowed = {str(path.relative_to(ROOT)) for path in files}
            assert set(staged) <= allowed, 'Unexpected existing staged files; leave source pending'
            subprocess.run(['git', 'commit', '-m', 'Report bounded Stage T434 offline relay outcomes'],
                           cwd=ROOT, check=True, timeout=30)
            call('transport.py', ['source_sync'], p.get('source_seconds', 180), cut['all'] - 5,
                 DOC / 'SOURCE.log', 'OPTIONAL_EXACT_SOURCE_SYNC', heartbeat=False)
            source = 'EXACT_REMOTE_ROOT_VERIFIED'
        except BaseException:
            save(DOC / 'SOURCE_PENDING.json', dict(error=traceback.format_exc(), epoch=time.time(), science_not_restarted=True))
        signal.alarm(0)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        print(json.dumps(dict(state=state, reason=reason, all_native_stopped=closed,
            six_gate=six_gate, eight_started=eight_started, remote=remote, source=source)), flush=True)


if __name__ == '__main__':
    main()
