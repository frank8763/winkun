"""Explicit new authorization; one bounded launch, audit and raw export."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT/'stage/runs/t661_value_state_retention'
DOC = ROOT/'stage/docs/day_20261120/t661'
import t661_wait as Wait


def read(file):
    return json.loads(Path(file).read_text())


def save(file, value):
    Path(file).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(file)+'.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(file)


def sha(file):
    import hashlib
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def register():
    assert not OUT.exists()
    p, auth = read(HERE/'protocol.json'), read(DOC/'AUTHORIZATION.json')
    assert auth['approved'] and not p['authorization_approved']
    for key in ['max_physics', 'max_commands', 'main_seconds', 'max_bytes', 'all_seconds', 'Actor_each_cap', 'Value_each_cap']:
        assert auth[key] == p[key], key
    assert read(DOC/'PREFLIGHT.json')['state'] == 'PASS_ZERO_NATIVE_ZERO_OPTIMIZER'
    for file, expected in p['inputs'].items():
        assert sha(file) == expected, file
    save(DOC/'RESOURCE_PREFLIGHT.json', Wait.Raw.resources())
    now = time.time()
    p.update(authorization_approved=True, registration_epoch=now,
        absolute_learning_stop=now+900, absolute_native_stop=now+6000, absolute_all_stop=now+p['all_seconds'])
    for file in [DOC/'AUTHORIZATION.json', DOC/'PREFLIGHT.json']:
        p['inputs'][str(file)] = sha(file)
    OUT.mkdir()
    (OUT/'episodes').mkdir()
    save(HERE/'protocol.json', p)
    save(OUT/'protocol.json', p)
    save(DOC/'REGISTERED.json', dict(state='REGISTERED_NOT_LAUNCHED', epoch=now,
        absolute_learning_stop=p['absolute_learning_stop'], absolute_native_stop=p['absolute_native_stop'],
        absolute_all_stop=p['absolute_all_stop']))
    print('T661_REGISTERED_NOT_LAUNCHED', p['absolute_all_stop'])


def launch():
    p = read(OUT/'protocol.json')
    assert p['authorization_approved'] and not (OUT/'launch.json').exists()
    Wait.Raw.resources()
    receipt = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    left = int(p['absolute_all_stop']-time.time()-2)
    assert left > p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p['source_seconds']+60
    cmd = ['systemd-run', '--user', '--unit=stage-t661', '--collect', '--property=Type=exec',
        '--property=CPUQuota=400%', '--property=CPUWeight=10', '--property=Nice=19', '--property=AllowedCPUs=8 10 12 14',
        '--property=MemoryMax=4G', '--property=MemorySwapMax=0', '--property=KillMode=control-group', '--property=TimeoutStopSec=5',
        f'--property=RuntimeMaxSec={left}', f'--property=WorkingDirectory={ROOT}',
        f'--property=StandardOutput=append:{OUT}/driver.log', '--property=StandardError=inherit', '--setenv=CUDA_VISIBLE_DEVICES=',
        'systemd-inhibit', '--what=sleep:idle', '--mode=block', '--who=Stage T661', '--why=Bounded previous-block baseline comparison',
        str(ROOT/'run_stage.sh'), str(HERE/'t661_control.py'), 'driver']
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    with (OUT/'launch.json').open('x') as handle:
        json.dump(dict(epoch=time.time(), unit='stage-t661', absolute_all_stop=p['absolute_all_stop'], source_remote=receipt['remote_commit']), handle)
    print(result.stderr.strip())


def call(name, seconds, children):
    with (OUT/(name+'.log')).open('xb') as log:
        child = subprocess.Popen([str(ROOT/'run_stage.sh'), str(HERE/'runtime.py'), name], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        children.append(child.pid)
        save(OUT/'status.json', dict(state='ACTIVE', action=name, child_pid=child.pid, driver_pid=os.getpid(), epoch=time.time()))
        try:
            child.wait(timeout=seconds)
            assert child.returncode == 0, (name, child.returncode)
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


def driver():
    p = read(OUT/'protocol.json')
    assert p['authorization_approved'] and sha(HERE/'protocol.json') == sha(OUT/'protocol.json')
    assert not (OUT/'DRIVER_STARTED.json').exists()
    receipt = read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    assert receipt['root_tree'] == subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    for file, expected in p['inputs'].items():
        assert sha(file) == expected, file
    assert time.time()+p['main_seconds']+p['audit_seconds']+p['backup_seconds']+p['source_seconds']+60 < p['absolute_all_stop']
    os.sched_setaffinity(0, {14})
    save(OUT/'DRIVER_STARTED.json', dict(pid=os.getpid(), epoch=time.time(), source_remote=receipt['remote_commit'], science_network=False))
    log = (OUT/'affinity.log').open('xb')
    helper = subprocess.Popen(['/usr/bin/python3', str(HERE/'affinity_guard.py')], cwd=ROOT, stdout=log, stderr=log)
    children, error = [], None
    started = time.time()
    try:
        until = time.time()+5
        while not (DOC/'AFFINITY_GUARD.json').exists() and time.time() < until:
            assert helper.poll() is None
            time.sleep(.05)
        assert read(DOC/'AFFINITY_GUARD.json')['pid'] == helper.pid
        call('main', p['main_seconds'], children)
        science_seconds = time.time()-started
        call('audit', p['audit_seconds'], children)
        assert read(OUT/'audit.json')['state'] == 'PASS'
    except BaseException:
        error = traceback.format_exc()
        save(OUT/'failure.json', dict(error=error, raw_preserved=True, no_restart=True))
        science_seconds = time.time()-started
    if (OUT/'worker_generations.json').exists():
        for group in read(OUT/'worker_generations.json'):
            children.extend([group['parent'], *group['workers']])
    stopped = all(not Path('/proc', str(pid)).exists() for pid in children)
    save(OUT/'SEALED.json', dict(all_scientific_processes_stopped=stopped, scientific_pids=sorted(set(children)), epoch=time.time()))
    (OUT/'AFFINITY_STOP').touch()
    try:
        helper.wait(timeout=5)
    except subprocess.TimeoutExpired:
        helper.terminate()
        helper.wait(timeout=3)
    log.close()
    summary = read(OUT/'summary.json') if (OUT/'summary.json').exists() else None
    calls = [json.loads(line) for line in (OUT/'optimizer_calls.jsonl').read_text().splitlines()] if (OUT/'optimizer_calls.jsonl').exists() else []
    optimizer_attempts = {who: {kind: sum(row[:4] == [who, 0, kind, 'BEFORE'] for row in calls)
        for kind in ['actor', 'value']} for who in ['live', 'shadow', 'independent_history']}
    assert all(v['actor'] <= p['Actor_each_cap'] and v['value'] <= p['Value_each_cap'] for v in optimizer_attempts.values())
    report = dict(state='FAILED' if error else 'COMPLETE', error=error, summary=summary,
        main_seconds= science_seconds, all_scientific_processes_stopped=stopped,
        optimizer_attempts=optimizer_attempts,
        remote='REMOTE_PENDING_LOCAL_RAW_RETAINED', epoch=time.time())
    save(OUT/'FINAL_REPORT.json', report)
    if stopped and time.time()+p['backup_seconds']+p['source_seconds']+30 < p['absolute_all_stop']:
        try:
            with (OUT/'backup.log').open('xb') as log:
                result = subprocess.run([str(ROOT/'run_stage.sh'), str(HERE/'transport.py')], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT, timeout=p['backup_seconds'])
            assert result.returncode == 0
            report['remote'] = 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
        except BaseException:
            report['backup_error'] = traceback.format_exc()
    save(OUT/'FINAL_REPORT_FINAL.json', report)
    save(DOC/'FINAL_REPORT.json', report)
    save(OUT/'status.json', dict(state=report['state'], all_science_stopped=stopped, epoch=time.time()))
    if stopped:
        status_file = ROOT/'stage/docs/RESEARCH_STATUS.md'
        status_file.write_text('2026-11-20 T661保留上一批Value单变量实验结束：'+report['state']+'，remote='+report['remote']+
            '；历史零Value对照复用非新样本；实际旧ID保持/场地新增/条件407及独立审计见day_20261120/t661/FINAL_REPORT。'
            '缺测null，非终身学习或泛化通过；未通过原384保持不得晋升。\n\n'+status_file.read_text())
        try:
            paths = [str(DOC.relative_to(ROOT)), 'stage/docs/RESEARCH_STATUS.md']
            paths.extend(str(file.relative_to(ROOT)) for file in (ROOT/'stage/docs/long_distance_20261101').glob('*T661*.json'))
            subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True, capture_output=True)
            staged = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
            if staged.returncode == 1:
                subprocess.run(['git', 'commit', '-m', 'Report bounded previous-block Value experiment'], cwd=ROOT, check=True, capture_output=True)
            if time.time()+p['source_seconds']+10 < p['absolute_all_stop']:
                subprocess.run([str(ROOT/'run_stage.sh'), str(ROOT/'stage/experiments/t629_offline_night_relay/t629_transport.py'), 'source_sync'],
                    cwd=ROOT, check=True, capture_output=True, timeout=p['source_seconds'], env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        except BaseException:
            save(DOC/'SOURCE_PENDING.json', dict(error=traceback.format_exc()))
    print(json.dumps(dict(state=report['state'], remote=report['remote'])), flush=True)


if __name__ == '__main__':
    {'register': register, 'launch': launch, 'driver': driver}[sys.argv[1]]()
