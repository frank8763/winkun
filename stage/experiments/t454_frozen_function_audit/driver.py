"""One frozen audit with bounded evidence transport; never launches training."""
from pathlib import Path
import hashlib, json, os, signal, subprocess, time, traceback
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'stage/runs/t454_frozen_function_audit'
DOC = ROOT/'stage/docs/review_20261110'


def read(f): return json.loads(f.read_text())
def save(f, x): f.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')


def child(*args, seconds):
    c = subprocess.Popen(['taskset', '-c', '0', str(ROOT/'run_stage.sh'), str(HERE/args[0]), *args[1:]],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        b, _ = c.communicate(timeout=max(1, seconds))
        if c.returncode:
            raise RuntimeError(b.decode(errors='replace')[-6000:])
    except BaseException:
        try: os.killpg(c.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: c.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(c.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            c.wait()
        raise


def main():
    p = read(OUT/'protocol.json')
    end = read(OUT/'launch.json')['all_stop']
    save(OUT/'driver_started.json', dict(pid=os.getpid(), epoch=time.time()))
    try:
        child('run.py', 'main', seconds=min(p['main_seconds'], end-time.time()-1300))
        child('run.py', 'audit', seconds=min(p['audit_seconds'], end-time.time()-700))
        s, a = read(OUT/'summary.json'), read(OUT/'audit.json')
        ids = [s['pid'], a['pid']]
        assert all(not Path('/proc', str(pid)).exists() for pid in ids)
        save(OUT/'SEALED.json', dict(all_scientific_processes_stopped=True, scientific_pids=ids, epoch=time.time()))
        files = [dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest())
            for f in sorted(OUT.rglob('*')) if f.is_file() and f.suffix not in ['.partial', '.lock', '.pyc']]
        save(DOC/'T454_LOCAL_MANIFEST.json', dict(files=files, remote='PENDING'))
        report = dict(state='COMPLETE_AUDITED_LOCAL', summary=s, audit=a, remote='PENDING',
            upstream_T453_remote='PENDING', six_gate=False, eight_started=False)
        save(DOC/'T454_FINAL_REPORT.json', report)
        try:
            child('transport.py', 'T454', seconds=min(p['backup_seconds'], end-time.time()-180))
            report['remote'] = read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T454.json')['state']
        except BaseException:
            save(DOC/'T454_BACKUP_ERROR.json', dict(error=traceback.format_exc(), original_raw_preserved=True))
        save(DOC/'T454_FINAL_REPORT.json', report)
        f = ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text(f"2026-11-10 T454四冻结模型同history函数审计已结束，0native/0优化器，独立audit={a['state']}，remote={report['remote']}；仅编码GRU/动作头双顺序代数分解，不是闭环因果/混合控制器。原328研究模型保留、T453仍未晋升/原备份PENDING，six门未过/8未进/Phase隔离。详review_20261110/T454_FINAL_REPORT及raw findings。\n\n"+f.read_text())
        subprocess.run(['git', 'add', 'stage/docs/review_20261110', 'stage/docs/long_distance_20261101', 'stage/docs/RESEARCH_STATUS.md'], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['git', 'commit', '-m', 'Report frozen Stage recurrent function and block audit'], cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try: child('transport.py', 'source_sync', seconds=min(150, end-time.time()-5))
        except BaseException: save(DOC/'T454_SOURCE_SYNC_ERROR.json', dict(error=traceback.format_exc(), local_committed=True))
    except BaseException:
        save(DOC/'T454_DRIVER_ERROR.json', dict(error=traceback.format_exc(), native=0, optimizers=0, no_restart=True))


if __name__ == '__main__':
    main()
