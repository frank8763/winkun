"""Single finite paired-control driver; scientific work does not need network."""
from pathlib import Path
import json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t443_sampling_control';DOC=ROOT/'stage/docs/research_rescue_20261109'
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
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('T443 supervisor stop')))
    save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end))
    try:
        run('run.py','main',seconds=min(p['main_seconds']+10,end-time.time()-2000))
        run('run.py','audit',seconds=min(p['audit_seconds'],end-time.time()-850))
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS'
        ids=[s['pid'],a['pid']]
        for arm in p['arms']:ids+=read(OUT/arm/'workers.json')['workers']
        assert all(not Path('/proc',str(pid)).exists() for pid in ids)
        report=dict(state='COMPLETE_AUDITED_LOCAL',physics=s['physics'],commands=s['commands'],main_seconds=s['elapsed_s'],
            arms={arm:dict(development=v['development'],rounds_completed=len(v['rounds']),
                live_Actor_Adam=sum(r['fit']['Actor_Adam'] for r in v['rounds']),live_V_Adam=sum(r['fit']['V_Adam'] for r in v['rounds']),
                physics=v['physics'],commands=v['commands']) for arm,v in s['results'].items()},
            audit_discarded_historical_Actor_plus_V_Adam=a['discarded_historical_Actor_plus_V_Adam'],
            cross_arm_identical_training_count=a['cross_arm_identical_training_count'],
            all_native_and_audit_processes_stopped=True,remote='PENDING',fresh16=None,reload78=None,six_gate=False,eight_started=False,
            storage=read(OUT/'storage_summary.json'),storage_audit='PASS',finished_epoch=time.time())
        save(DOC/'T443_FINAL_REPORT.json',report)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,epoch=time.time()))
        # Do not write to OUT, including its console, after sealing.
        try:
            run('transport.py','T443',seconds=min(650,end-time.time()-220))
            report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T443.json')['state']
        except BaseException:save(DOC/'T443_BACKUP_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True))
        save(DOC/'T443_FINAL_REPORT.json',report)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text(f"2026-11-09 T443困难起态分配对照已结束/独立历史Actor+V Adam重构PASS，physics={report['physics']} commands={report['commands']}，两固定最终模型完整329见research_rescue_20261109/T443_FINAL_REPORT.json；跨臂相同轨迹{report['cross_arm_identical_training_count']}明确物理重复非独立。fresh16/reload78未准入/six_gate=False/8未起，研究副本不晋升，remote={report['remote']}。无损冷存原字节本地gzip/旧完整远端保留，用t443_sampling_control/reserve.py load_original解码，不能读取已编码旧裸路径。Phase隔离/safeT255不改，不重启或扩大旧预算。\n\n"+status.read_text())
        subprocess.run(['git','add','stage/docs/research_rescue_20261109','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        subprocess.run(['git','commit','-m','Report paired Stage task allocation control and reversible evidence reserve'],cwd=ROOT,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        run('transport.py','source_sync',seconds=min(210,end-time.time()-5))
    except BaseException:save(DOC/'T443_DRIVER_ERROR.json',dict(error=traceback.format_exc(),epoch=time.time(),raw_preserved=True,no_restart=True))
if __name__=='__main__':main()
