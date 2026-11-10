"""Frozen evaluation finalizer; never starts or resumes learning."""
from pathlib import Path
import hashlib, json, os, signal, subprocess, time, traceback
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t459_nonlinear_residual_repair'
DOC = ROOT/'stage/docs/review_20261110'


def read(f): return json.loads(f.read_text())
def save(f, x): f.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def child(script, *args, seconds):
    c = subprocess.Popen(['taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/script), *args],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        b, _ = c.communicate(timeout=max(1, seconds))
        if c.returncode:
            raise RuntimeError(b.decode(errors='replace')[-6500:])
    except BaseException:
        try: os.killpg(c.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: c.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try: os.killpg(c.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            c.wait()
        raise


def main():
    p = read(OUT/'protocol.json')
    end = read(OUT/'launch.json')['all_stop']
    save(OUT/'driver_started.json', dict(pid=os.getpid(), epoch=time.time(), all_stop=end))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(TimeoutError('T459 hardstop')))
    try:
        try: child('run.py', 'main', seconds=min(p['main_seconds']+5, end-time.time()-1100))
        except BaseException: save(DOC/'T459_MAIN_ERROR.json', dict(error=traceback.format_exc(), no_restart=True))
        s = read(OUT/'summary.json')
        generations = read(OUT/'worker_generations.json') if (OUT/'worker_generations.json').exists() else []
        ids = [s['pid']]+[v for g in generations for v in g['workers']]
        assert all(not Path('/proc', str(pid)).exists() for pid in ids)
        audit = None
        if s['state'] == 'COMPLETE':
            try:
                child('run.py', 'audit', seconds=min(p['audit_seconds'], end-time.time()-800))
                audit = read(OUT/'audit.json')
                ids.append(audit['pid'])
            except BaseException: save(DOC/'T459_AUDIT_ERROR.json', dict(error=traceback.format_exc(), no_repeat=True))
        assert all(not Path('/proc', str(pid)).exists() for pid in ids)
        save(OUT/'SEALED.json', dict(all_scientific_processes_stopped=True, scientific_pids=ids, epoch=time.time()))
        files = [dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.lock', '.partial', '.pyc']]
        save(DOC/'T459_LOCAL_MANIFEST.json', dict(files=files, remote='PENDING'))
        report = dict(state='COMPLETE_AUDITED_LOCAL' if audit else 'STOPPED_NOT_FULL_AUDIT_PASS',
            summary=s, audit=audit, remote='PENDING', six_gate=False, eight_started=False,
            neural_residual_supervised_pilot_NOT_new_PPO=True)
        save(DOC/'T459_FINAL_REPORT.json', report)
        try:
            child('transport.py', 'T459', seconds=min(620, end-time.time()-200))
            report['remote'] = read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T459.json')['state']
        except BaseException: save(DOC/'T459_BACKUP_ERROR.json', dict(error=traceback.format_exc(), raw_preserved=True))
        save(DOC/'T459_FINAL_REPORT.json', report)
        c = s['candidate']
        result = '完整329未完成' if c is None else f"宽{c['wide54']['success']}/54旧{c['old275']['success']}/275，development_admitted={c['development_admitted']}"
        f = ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text(f"2026-11-10 T459原策略冻结非线性神经修正已结束，state={report['state']}，remote={report['remote']}；{result}；128BC上限/0新PPO和教师，physics={s['physics']} commands={s['commands']}。固定末候选完整329，新增1074残差参数/原52866冻结，capacity+freshAdam联合pilot非原创或单因素因果。16fresh/78未预留未执行null，six_gateFalse/8未进，原328/safeT255保留/Phase隔离。详review_20261110/T459_FINAL_REPORT。\n\n"+f.read_text())
        subprocess.run(['git', 'add', 'stage/docs/review_20261110', 'stage/docs/long_distance_20261101', 'stage/docs/RESEARCH_STATUS.md'], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['git', 'commit', '-m', 'Report frozen-base neural residual Stage repair'], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try: child('transport.py', 'source_sync', seconds=min(180, end-time.time()-5))
        except BaseException: save(DOC/'T459_SOURCE_SYNC_ERROR.json', dict(error=traceback.format_exc(), local_committed=True))
    except BaseException:
        save(DOC/'T459_DRIVER_ERROR.json', dict(error=traceback.format_exc(), raw_preserved=True, no_restart=True, epoch=time.time()))


if __name__ == '__main__':
    main()
