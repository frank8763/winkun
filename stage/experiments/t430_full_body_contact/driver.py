"""Single bounded finalizer for the zero-step Stage contact analysis."""
from pathlib import Path
import importlib.util
import os
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stage430_geometry_contract', HERE / 'run.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def call(file, args, seconds):
    p = C.read(C.OUT / 'protocol.json')
    limit = min(seconds, p['deadline_all'] - time.time() - 3)
    assert limit > 0
    with (C.DOC / (file.replace('.py', '') + '_' + (args[0] if args else 'main') + '.log')).open('xb') as log:
        child = subprocess.Popen(['taskset', '-c', '0', str(C.ROOT / 'run_stage.sh'),
            str(HERE / file), *args], cwd=C.ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        if not (file == 'transport.py' and args == ['source_sync']):
            C.save(C.DOC / 'DRIVER_STATUS.json', dict(state=file, args=args, epoch=time.time(),
                 driver_pid=os.getpid(), child_pid=child.pid, zero_physics=True))
        try:
            code = child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, 15)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, 9)
                child.wait()
            raise
        assert code == 0, (file, args, code)
        assert not Path('/proc', str(child.pid)).exists()


def main():
    p = C.read(C.OUT / 'protocol.json')
    C.verify(p)
    C.save(C.OUT / 'driver_started.json', dict(pid=os.getpid(), epoch=time.time()))
    try:
        call('run.py', ['main'], p['main_seconds'])
        assert C.read(C.OUT / 'summary.json')['state'] == 'COMPLETE'
        call('audit.py', [], p['audit_seconds'])
        assert C.read(C.OUT / 'audit.json')['state'] == 'PASS'
        C.save(C.OUT / 'SEALED.json', dict(all_scientific_processes_stopped=True,
             physics=0, Actor_Adam=0, epoch=time.time()))
        members = [dict(path=str(f.relative_to(C.ROOT)), bytes=f.stat().st_size, sha256=C.sha(f))
                   for f in sorted(C.OUT.rglob('*')) if f.is_file()]
        C.save(C.OUT / 'local_manifest.json', dict(files=members, zero_physics=True))
        remote = 'REMOTE_PENDING'
        try:
            call('transport.py', ['T430'], p['backup_seconds'])
            receipt = C.read(C.ROOT / 'stage/docs/long_distance_20261101/release_receipt_T430.json')
            assert receipt['state'] == 'FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
            remote = receipt['state']
        except BaseException:
            C.save(C.DOC / 'REMOTE_ERROR.json', dict(error=traceback.format_exc(), originals_preserved=True))
        result = C.read(C.OUT / 'contact_results.json')
        C.save(C.DOC / 'FINAL_REPORT.json', dict(state='COMPLETE_STATIC_CONTACT_AUDITED',
             physics=0, Actor_Adam=0, remote=remote, results=result,
             no_navigation_training=True, six_gate=False, eight_started=False,
             source_sync='Check latest_source_sync_receipt.json; this field is not a sync claim.'))
        rows = ['# T430 完整碰撞体静态核查', '',
                '0物理积分、0 Actor/Adam；不是新的导航或恢复实验。独立静态重建 audit PASS。', '',
                '| 已有轨迹终态 | 静态非地面接触 |', '| --- | --- |']
        for row in result['rows']:
            contacts = [f"{v['name1']} / {v['name2']} ({v['distance_m']:.7f}m)"
                        for v in row['obstacle_contacts'] if v['distance_m'] <= .002]
            rows.append('| ' + row['label'] + ' | ' + ('; '.join(contacts) or '无') + ' |')
        rows += ['', '只核查保存的终态几何；没有重放原生求解器上一积分时刻的contact，'
                 '不把当前静态查询当独立导航验收或可恢复性证明。模型按当前factory MJB及T326完整静态碰撞几何核对，'
                 '不声称重新构造所有运行时动力学参数。', '', '远端原始备份状态：' + remote + '。']
        (C.DOC / 'FINAL_REPORT.md').write_text('\n'.join(rows) + '\n')
        status = C.ROOT / 'stage/docs/RESEARCH_STATUS.md'
        prefix = ('2026-11-08 T430已完成零物理完整碰撞体静态终态核查/独立auditPASS，'
                  '0ActorAdam/0新导航，原始contact_results与报告见contact_20261108；'
                  '不是原生contact历史重放/可恢复性或六米验收证明。'
                  '研究Actor未变、six完整门未过/8未开始；safeT255保持/窄暂停/Phase隔离。\n\n')
        status.write_text(prefix + status.read_text())
        C.save(C.DOC / 'DRIVER_STATUS.json', dict(state='COMPLETE_STATIC_CONTACT_AUDITED',
             epoch=time.time(), driver_pid=os.getpid(), remote=remote, physics=0, Actor_Adam=0,
             source_sync='Verify latest_source_sync_receipt.json; no premature sync claim.'))
        subprocess.run(['git', 'add', str(HERE.relative_to(C.ROOT)),
                        str(C.DOC.relative_to(C.ROOT)), 'stage/docs/feedback_20261108/NEXT_CONTROL_PRIMARY_SOURCES.md',
                        'stage/docs/RESEARCH_STATUS.md'], cwd=C.ROOT, check=True)
        for name in ['release_receipt_T430.json', 'source_manifest_T430.json', 'T430_small_parts_progress.json']:
            file = C.ROOT / 'stage/docs/long_distance_20261101' / name
            if file.exists():
                subprocess.run(['git', 'add', str(file.relative_to(C.ROOT))], cwd=C.ROOT, check=True)
        subprocess.run(['git', 'diff', '--cached', '--check'], cwd=C.ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Audit complete collision geometry at existing failed endpoints'],
                       cwd=C.ROOT, check=True)
        call('transport.py', ['source_sync'], p['source_seconds'])
    except BaseException:
        C.save(C.DOC / 'DRIVER_ERROR.json', dict(error=traceback.format_exc(),
             epoch=time.time(), original_evidence_preserved=True, no_restart=True))
        raise


if __name__ == '__main__':
    main()
