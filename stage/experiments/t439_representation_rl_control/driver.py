"""Unique local finalizer for a single bounded RL mechanism."""
from pathlib import Path
import subprocess,sys,json,time,traceback,os
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t439_representation_rl_control';DOC=ROOT/'stage/docs/review_20261109'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def run(script,*args,timeout):
    subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,check=True,timeout=timeout)
def main():
    p=read(OUT/'protocol.json');start=time.time();save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start))
    try:
        run('run.py','representation_rl',timeout=min(3650,p['deadline_all']-time.time()-900))
        run('audit.py','representation_rl',timeout=min(950,p['deadline_all']-time.time()-900))
        s=read(OUT/'fresh_rl/summary.json');a=read(OUT/'fresh_rl/audit.json');assert a['state']=='PASS'
        pids=[read(OUT/'fresh_rl/main_started.json')['pid']]+read(OUT/'fresh_rl/workers.json')['workers'];assert all(not Path('/proc',str(pid)).exists() for pid in pids)
        result=s['result'];development=read(result['check_file']) if result['check_file'] else None
        report=dict(state='COMPLETE_AUDITED_LOCAL',finished_epoch=time.time(),elapsed_s=time.time()-start,
            main_seconds=s['elapsed_s'],physics=s['physics'],commands=s['commands']+8,
            training_unique=sum(1 for line in (OUT/'fresh_rl/cases.jsonl').read_text().splitlines() if read_line(line)['role'].startswith('TRAIN_')),
            live=result['updates']['live'],shadow=result['updates']['shadow'],research_rounds_kept=result['retained_research_rounds'],
            final_accepted=result['accepted'],full329_executed=result['full329_executed'],
            old275=development['old275'] if development else None,wide54=development['wide54'] if development else None,
            fresh16=None,reload78=None,six_gate=False,eight_started=False,
            all_scientific_processes_stopped=True,independent_audit='PASS',remote='PENDING',
            old_T422_history_audit_NOT_repaired=True,safeT255_not_promoted=True)
        save(DOC/'T439_FINAL_REPORT.json',report);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,epoch=time.time()))
        # No file inside OUT is written after sealing: archive covers all raw.
        try:
            run('transport.py','T439',timeout=min(650,p['deadline_all']-time.time()-240))
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T439.json')['state'];save(DOC/'T439_FINAL_REPORT.json',report)
        except BaseException:save(DOC/'T439_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,scientific_run_not_restarted=True))
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        entry=f"2026-11-09 T439唯一新闭环RL已结束，live={report['live']}/shadow={report['shadow']}，研究保留轮数={report['research_rounds_kept']}，full329={report['full329_executed']}，final_accept={report['final_accepted']}；fresh16/reload78/8均未执行，six稳定门未宣称。全部科学进程退出/独立auditPASS，remote={report['remote']}。详review_20261109/T439_FINAL_REPORT.json，不重启/扩大已结束目录，safeT255不晋升，Phase隔离。\n\n"
        status.write_text(entry+status.read_text())
        subprocess.run(['git','add','stage/docs/review_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Report bounded Stage joint closed-loop RL pilot'],cwd=ROOT,check=True)
        run('transport.py','source_sync',timeout=min(240,p['deadline_all']-time.time()-5))
    except BaseException:
        save(DOC/'T439_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),no_training_restart=True,raw_preserved=True));raise
def read_line(s):return json.loads(s)
if __name__=='__main__':main()
