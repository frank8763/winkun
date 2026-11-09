"""Exactly one last candidate, then report and stop; no fallback branches."""
from pathlib import Path
import subprocess,json,sys,time,os,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t440_final_functional_constraint';DOC=ROOT/'stage/docs/review_20261109'
def read(f):return json.loads(Path(f).read_text())
def save(f,x):Path(f).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def call(mode,seconds):subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'run.py'),mode],cwd=ROOT,check=True,timeout=seconds)
def main():
    p=read(OUT/'protocol.json');save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time()))
    try:
        call('fit',320)
        fit=read(OUT/'fit_summary.json')
        if fit['native_allowed']:call('native',min(2400,p['deadline_all']-time.time()-900))
        call('audit',min(320,p['deadline_all']-time.time()-600))
        n=read(OUT/'native_summary.json') if (OUT/'native_summary.json').exists() else None
        result=n['result'] if n else dict(wide54=None,old275=None,fresh16=None,reload78=None,six_gate=False,eight_started=False)
        success=result['six_gate'];report=dict(state='COMPLETE_AUDITED',offline_fit=fit['fit'],native=n,
            result=result,all_scientific_processes_stopped=True,remote='PENDING',
            line_stopped_by_user_condition=not success,no_automatic_new_branches=True,
            final_model=fit['proposed_model'] if success else fit['initial_model'],rejected_model_preserved=not success)
        save(DOC/'T440_FINAL_REPORT.json',report);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,epoch=time.time()))
        try:
            subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T440'],cwd=ROOT,check=True,timeout=min(600,p['deadline_all']-time.time()-180))
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T440.json')['state'];save(DOC/'T440_FINAL_REPORT.json',report)
        except BaseException:save(DOC/'T440_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        verdict='六米完整门通过，未自动扩距' if success else '最后尝试未通过，按用户条件停止Stage实验线'
        text=f"2026-11-09 T440 {verdict}。仅一个功能约束策略提案，offline={fit['fit']['state']}，native={'EXECUTED' if n else 'NOT_EXECUTED'}，six_gate={success}；独立auditPASS、remote={report['remote']}。不追加分支/轮数/seed，不重启旧作业，safeT255保留，Phase隔离。详review_20261109/T440_FINAL_REPORT.json。\n\n"
        status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text(text+status.read_text())
        subprocess.run(['git','add','stage/docs/review_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Report last bounded Stage functional-constraint attempt and stop verdict'],cwd=ROOT,check=True)
        subprocess.run(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,check=True,timeout=170)
    except BaseException:
        save(DOC/'T440_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),no_training_restart=True,line_stop_pending_evidence=True));raise
if __name__=='__main__':main()
