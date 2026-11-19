"""One explicitly authorized hour, no training, no automatic retry."""
from pathlib import Path
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
OUT = ROOT / 'stage/runs/t659_continuation_gate'
DOC = ROOT / 'stage/docs/day_20261119/t659'
sys.path.insert(0, str(ROOT / 'stage/experiments/t645_iterative_recovery_relay'))
import control as Resource


def read(f):
    return json.loads(Path(f).read_text())


def save(f, value):
    Path(f).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def register():
    assert not OUT.exists()
    p, a = read(HERE/'protocol.json'), read(DOC/'AUTHORIZATION.json')
    assert a['approved'] and not p['authorization_approved']
    assert a['protocol_revision'] == p['protocol_revision'] == 2
    for key in ['all_seconds','max_physics','max_commands','max_bytes','global_bytes','new_Actor','new_Critic','new_optimizer']:
        assert a[key] == p[key], key
    assert read(DOC/'PREFLIGHT.json')['state'] == 'PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for f, h in p['inputs'].items():
        assert sha(f) == h, f
    save(DOC/'RESOURCE_PREFLIGHT.json', Resource.resources())
    now = time.time()
    p.update(authorization_approved=True, preparation_only=False, registration_epoch=now,
             absolute_all_stop=now+p['all_seconds'], absolute_native_stop=now+1900,
             absolute_learning_stop=now,
             user_authorization='User approved the T659 one-hour frozen continuation panel and existing group balances; see AUTHORIZATION.json')
    p['inputs'].update({str(DOC/'AUTHORIZATION.json'): sha(DOC/'AUTHORIZATION.json'),
                        str(DOC/'PREFLIGHT.json'): sha(DOC/'PREFLIGHT.json')})
    OUT.mkdir()
    (OUT/'episodes').mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    save(DOC/'REGISTERED.json', dict(state='REGISTERED_NOT_LAUNCHED', epoch=now,
                                   all_stop=p['absolute_all_stop'], native_stop=p['absolute_native_stop']))
    print('T659_REGISTERED_NOT_LAUNCHED', p['absolute_all_stop'])


def execute(action, seconds, children):
    script = 'transport.py' if action == 'backup' else 't659_runtime.py'
    with (OUT/(action+'.log')).open('xb') as log:
        c = subprocess.Popen([str(ROOT/'run_stage.sh'), str(HERE/script), action], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        children.append(c.pid)
        try:
            c.wait(timeout=seconds)
            assert c.returncode == 0, (action, c.returncode)
        except BaseException:
            try:
                os.killpg(c.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                c.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(c.pid, signal.SIGKILL)
                c.wait()
            raise


def driver():
    p = read(OUT/'protocol.json')
    assert p['authorization_approved'] and not (OUT/'DRIVER_STARTED.json').exists()
    assert (HERE/'protocol.json').read_bytes() == (OUT/'protocol.json').read_bytes()
    receipt = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git','rev-parse','HEAD^{tree}'], cwd=ROOT, text=True).strip()
    for f, h in p['inputs'].items():
        assert sha(f) == h, f
    assert time.time()+p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p['source_seconds']+60 < p['absolute_all_stop']
    Resource.resources()
    os.sched_setaffinity(0, {14})
    save(OUT/'DRIVER_STARTED.json', dict(pid=os.getpid(), epoch=time.time(), source_remote=receipt['remote_commit']))
    log = (OUT/'affinity.log').open('xb')
    helper = subprocess.Popen(['/usr/bin/python3', str(HERE/'affinity_guard.py')], cwd=ROOT, stdout=log, stderr=log)
    children, error, main_used = [], None, 0.
    try:
        ready_until = time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time() < ready_until:
            assert helper.poll() is None
            time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid'] == helper.pid
        assert time.time()+p['main_seconds'] < p['absolute_native_stop']
        start = time.time()
        try:
            execute('main', p['main_seconds'], children)
        finally:
            main_used = time.time()-start
        assert helper.poll() is None
        execute('audit', p['audit_seconds'], children)
        assert read(OUT/'audit.json')['state'] == 'PASS'
    except BaseException:
        error = traceback.format_exc()
    if (OUT/'worker_generations.json').exists():
        for generation in read(OUT/'worker_generations.json'):
            children.extend([generation['parent'], *generation['workers']])
    children = sorted(set(children))
    gone = all(not Path('/proc', str(pid)).exists() for pid in children)
    if not gone:
        error = (error or '')+'\nScientific child still present; no backup or source sync permitted.'
    save(OUT/'SEALED.json', dict(all_scientific_processes_stopped=gone, scientific_pids=children, epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:
        helper.wait(timeout=5)
    except subprocess.TimeoutExpired:
        helper.terminate()
        try:
            helper.wait(timeout=2)
        except subprocess.TimeoutExpired:
            helper.kill()
            helper.wait(timeout=2)
    log.close()
    summary = read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    save(OUT/'LEDGER.json', dict(main_seconds=main_used, summary=summary, new_Actor=0, new_Critic=0, new_optimizer=0,
                                counts_complete=bool(summary),
                                physics_upper=summary['physics'] if summary else p['max_physics'],
                                commands_upper=summary['commands'] if summary else p['max_commands']))
    report = dict(state='FAILED' if error else 'COMPLETE', error=error, summary=summary,
                  audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,
                  all_scientific_processes_stopped=gone, remote='REMOTE_PENDING_LOCAL_RAW_RETAINED',
                  scope='Frozen per-query continuation replacement; no learning, predictor calibration or model promotion')
    save(OUT/'FINAL_REPORT.json', report)
    save(OUT/'local_manifest.json', dict(files=[dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['driver.log','local_manifest.json']]))
    if gone and time.time()+p['backup_seconds']+p['source_seconds']+30 < p['absolute_all_stop']:
        try:
            execute('backup', p['backup_seconds'], [])
            report['remote'] = 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:
            report['backup_error'] = traceback.format_exc()
    save(OUT/'FINAL_REPORT_FINAL.json', report)
    save(DOC/'FINAL_REPORT.json', report)
    save(OUT/'status.json', dict(state=report['state'], all_science_stopped=gone, epoch=time.time()))
    status = ROOT/'stage/docs/RESEARCH_STATUS.md'
    status.write_text('2026-11-19 T659冻结续行因果检验结束：'+report['state']+'，remote='+report['remote']+
        '；实际28新测量28重放/20历史复用panel及审计见day_20261119/t659/FINAL_REPORT；0学习，不是新导航成功或Critic通过，407null，T501保留。\n\n'+status.read_text())
    if not gone:
        save(DOC/'SOURCE_PENDING.json', dict(reason='Scientific children not all gone; preserve local evidence, no source sync'))
        return
    try:
        paths = [str(DOC.relative_to(ROOT)), 'stage/docs/RESEARCH_STATUS.md']
        paths.extend(str(f.relative_to(ROOT)) for f in (ROOT/'stage/docs/long_distance_20261101').glob('*T659*.json'))
        subprocess.run(['git','add','--',*paths], cwd=ROOT, check=True, capture_output=True)
        subprocess.run(['git','commit','-m','Report frozen candidate continuation replacement panel'], cwd=ROOT, check=True, capture_output=True)
        if time.time()+p['source_seconds']+10 < p['absolute_all_stop']:
            subprocess.run([str(ROOT/'run_stage.sh'), str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'), 'source_sync'],
                cwd=ROOT, check=True, capture_output=True, timeout=p['source_seconds'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    except BaseException:
        save(DOC/'SOURCE_PENDING.json', dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'], remote=report['remote'])))


def launch():
    p = read(OUT/'protocol.json')
    assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    receipt = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git','rev-parse','HEAD^{tree}'], cwd=ROOT, text=True).strip()
    Resource.resources()
    left = int(p['absolute_all_stop']-time.time()-2)
    assert left > p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p['source_seconds']+60
    cmd = ['systemd-run','--user','--unit=stage-t659','--collect','--property=Type=exec',
           '--property=CPUQuota=400%','--property=CPUWeight=10','--property=Nice=19',
           '--property=AllowedCPUs=8 10 12 14','--property=MemoryMax=4G','--property=MemorySwapMax=0',
           '--property=KillMode=control-group','--property=TimeoutStopSec=5',f'--property=RuntimeMaxSec={left}',
           f'--property=WorkingDirectory={ROOT}',f'--property=StandardOutput=append:{OUT}/driver.log',
           '--property=StandardError=inherit','--setenv=CUDA_VISIBLE_DEVICES=',
           'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage T659',
           '--why=Bounded frozen continuation replacement measurement',str(ROOT/'run_stage.sh'),str(HERE/'control.py'),'driver']
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with (OUT/'launch.json').open('x') as f:
        json.dump(dict(unit='stage-t659', epoch=time.time(), absolute_all_stop=p['absolute_all_stop'], source_remote=receipt['remote_commit']), f, indent=2)
    print(result.stderr.strip())


if __name__ == '__main__':
    {'register': register, 'launch': launch, 'driver': driver}[sys.argv[1]]()
