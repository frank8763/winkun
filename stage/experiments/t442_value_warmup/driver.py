"""Single finite V-only finalizer; no online Actor followup is registered."""
from pathlib import Path
import json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t442_value_warmup';DOC=ROOT/'stage/docs/research_rescue_20261109'
def read(f):return json.loads(Path(f).read_text())
def save(f,v):Path(f).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def run(script,*args,seconds):
    c=subprocess.Popen(['taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/script),*args],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        out,_=c.communicate(timeout=max(1,seconds))
        if c.returncode:raise RuntimeError(script+' exit '+str(c.returncode)+': '+out.decode(errors='replace')[-6000:])
    except BaseException:
        try:os.killpg(c.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:c.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:os.killpg(c.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            c.wait()
        raise

def main():
    p=read(OUT/'protocol.json');end=read(OUT/'launch.json')['all_stop']
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T442 supervisor stop')))
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    try:
        run('run.py','main',seconds=min(p['main_seconds']+10,end-time.time()-2000))
        run('run.py','audit',seconds=min(p['audit_seconds'],end-time.time()-900))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS'
        ids=[s['pid'],a['pid']];assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',native_steps=0,Actor_forwards=0,Actor_updates=0,
            live_V_Adam=256,shadow_V_Adam=256,audit_discarded_V_Adam=a['discarded_historical_V_Adam'],
            preflight_discarded_V_Adam=read(OUT/'preflight.json')['discarded_synthetic_V_Adam'],
            main_seconds=s['elapsed_s'],audit_seconds=a['elapsed_s'],screening=s['result']['screening'],
            grouped_validation_metrics={arm:s['result'][arm]['metrics']['groups']['validation'] for arm in ['GAE','MC']},
            zero_validation=s['result']['zero_metrics']['groups']['validation'],
            models={arm:s['result'][arm]['model'] for arm in ['GAE','MC']},
            remote='PENDING',six_gate=False,eight_started=False,online_actor_followup_started=False,finished_epoch=time.time())
        save(DOC/'T442_FINAL_REPORT.json',report)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        # All OUT writes end here; transport cannot mutate its evidence archive.
        try:
            run('transport.py','T442',seconds=min(650,end-time.time()-230))
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T442.json')['state']
        except BaseException:save(DOC/'T442_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        save(DOC/'T442_FINAL_REPORT.json',report)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text(f"2026-11-09 T442价值预热两目标对照已完整结束/独立256历史VAdam重构PASS；live256/shadow256，0native0Actorforward0Actorupdate。分组验证联合筛选={report['screening']['all_pass']}，remote={report['remote']}；仅预测诊断，不是导航通过/新onpolicyV证据，不接Actor、不自动后续。safeT255和T441研究模型SHA保持/Phase隔离。见research_rescue_20261109/T442_FINAL_REPORT.json，原始负结果全部保留，不重启/追加预算。\n\n"+status.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report Stage T442 grouped value warmup target control'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        run('transport.py','source_sync',seconds=min(220,end-time.time()-5))
    except BaseException:
        save(DOC/'T442_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),raw_preserved=True,no_training_restart=True))

if __name__=='__main__':main()
