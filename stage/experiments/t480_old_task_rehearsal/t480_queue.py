"""Finite, network-independent waiting; run registration only after T479 exits.

Waiting creates no run or periodic state files. The later cache/source/archive
steps require network; a failure stops with evidence, never retries training.
"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DOC = ROOT / 'stage/docs/review_20261111'
PREVIOUS = ROOT / 'stage/runs/t479_balanced_success_sil'
OUT = ROOT / 'stage/runs/t480_old_task_rehearsal'


def read(f):
    return json.loads(Path(f).read_text())


def previous_done():
    # Completion of a child or a native summary alone does not authorize us.
    if not (DOC / 'T479_FINAL_REPORT.json').exists():
        return False
    files = [PREVIOUS / 'launch.json', PREVIOUS / 'driver_started.json', PREVIOUS / 'SEALED.json']
    ids = []
    for f in files:
        if not f.exists():
            return False
        d = read(f)
        if d.get('pid'):
            ids.append(d['pid'])
        ids.extend(d.get('scientific_pids', []))
    return all(not Path('/proc', str(pid)).exists() for pid in ids)


def call(args, seconds):
    child = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             start_new_session=True)
    try:
        b, _ = child.communicate(timeout=seconds)
        if child.returncode:
            raise RuntimeError(b.decode(errors='replace')[-6000:])
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        raise


def commit(message):
    subprocess.run(['git', 'add', str(HERE.relative_to(ROOT)),
                    'stage/docs/review_20261111/T480_PLAN.md',
                    'stage/docs/review_20261111/T480_BUDGET.json',
                    'stage/docs/review_20261111/T480_READINESS.json'], cwd=ROOT, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(['git', 'commit', '-m', message], cwd=ROOT, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    wait_stop = datetime(2026, 11, 11, 14, 0, tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
    while not previous_done():
        assert time.time() < wait_stop, 'Previous writer did not finish before bounded queue admission stop'
        time.sleep(10)
    assert not OUT.exists()
    assert read(DOC / 'T479_FINAL_REPORT.json')['remote'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    sys.path.insert(0, str(HERE))
    import control as C
    C.prepare()
    C.save(OUT / 'queue_owner.json', dict(pid=os.getpid(), role='sequential_non_native_orchestrator',
           script_sha256=C.sha(HERE / 't480_queue.py'), epoch=time.time(),
           previous_full_driver_exited=True, no_parallel_native=True))
    C.save(DOC / 'T480_READINESS.json', dict(state='REGISTERED_NOT_PREFLIGHTED_NOT_LAUNCHED',
           previous_audit_and_raw_remote_complete=True))
    commit('Register finite old-task Vtrace rehearsal pilot with bounded cache plan')
    launcher = ['taskset', '-c', '0', str(ROOT / 'run_stage.sh')]
    call(launcher + [str(HERE / 'transport.py'), 'source_sync'], 180)
    call(launcher + [str(HERE / 'control.py'), 'compress'], 300)
    call(launcher + [str(HERE / 'control.py'), 'compression_audit'], 180)
    call(launcher + [str(HERE / 'control.py'), 'cache'], 1200)
    # A distinct process verifies cache ledger/markers; the parent only waits.
    call(launcher + [str(HERE / 'control.py'), 'cache_audit'], 180)
    call(launcher + [str(HERE / 'control.py'), 'preflight'], 300)
    C.save(DOC / 'T480_READINESS.json', dict(state='PREFLIGHT_PASS_NOT_LAUNCHED',
           previous_audit_raw_backup_and_writer_exit=True,
           cache=C.read(OUT / 'cache_audit.json'), preflight=C.read(OUT / 'preflight.json')))
    commit('Record independent cache and old-task Vtrace engineering preflight evidence')
    call(launcher + [str(HERE / 'transport.py'), 'source_sync'], 180)
    C.launch()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        # No claims of launch, no native retry, no source/protocol edits.
        target = OUT if OUT.exists() else DOC
        (target / 'T480_QUEUE_STOPPED.json').write_text(json.dumps(dict(
            state='STOPPED_NO_AUTOMATIC_RETRY', pid=os.getpid(), epoch=time.time(),
            error=traceback.format_exc(), raw_preserved=True), indent=2) + '\n')
        raise
