"""One diagnostic, independent audit, complete raw backup, then stop."""
from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t513_goal_feedback_teacher';DOC=ROOT/'stage/docs/day_20261113'
def read(f):return json.loads(Path(f).read_text())
def save(f,d):
    f=Path(f);tmp=Path(str(f)+'.partial');tmp.write_text(json.dumps(d,ensure_ascii=False)+'\n');tmp.replace(f)
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def main():
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),all_stop=end,source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False))
    def call(action,seconds):
        with (OUT/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t513_runtime.py'),action,'probe'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:c.wait(timeout=seconds);assert c.returncode==0,(action,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(c.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    c.wait()
                raise
    try:
        folder=OUT/'probe';folder.mkdir()
        save(folder/'protocol.json',dict(_root_sha256=sha(OUT/'protocol.json'),out=str(folder),phase='probe',max_physics=p['root_max_physics'],max_commands=6))
        save(OUT/'status.json',dict(state='RUNNING_REAL_STATE_RECOVERY',epoch=time.time()))
        call('main',603);s=read(folder/'summary.json');assert s['state']=='COMPLETE'
        ids=[s['pid']]+[i for g in read(folder/'worker_generations.json') for i in [g['parent']]+g['workers']]
        assert all(not Path('/proc',str(i)).exists() for i in ids)
        call('audit',300);audit=read(folder/'audit.json');assert audit['state']=='PASS';ids.append(audit['pid'])
        assert all(not Path('/proc',str(i)).exists() for i in ids)
        files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file()]
        save(folder/'local_manifest.json',dict(state='ALL_LOCAL_MEMBER_SHA_VERIFIED',files=files))
        assert all(sha(ROOT/f['path'])==f['sha256'] for f in files)
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(ids)),finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED',epoch=time.time()))
        remote='PENDING'
        try:
            b=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T513'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(600,end-time.time()-180))
            assert b.returncode==0,b.stdout.decode(errors='replace')[-2000:]
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T513.json')['state']
        except BaseException:save(DOC/'T513_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report=dict(state='COMPLETE_AUDIT_PASS',result=s['result'],accounting=dict(physics=s['physics'],commands=s['commands']+6,main_seconds=s['elapsed_s']),remote=remote,all_scientific_processes_stopped=True,epoch=time.time())
        save(DOC/'T513_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text('2026-11-13 T513固定目标反馈训练期教师验证结束，conditional='+str(s['result']['teacher_tally'])+'，qualified='+str(s['result']['qualified_teacher_successes'])+'，0Adam/0新PG/模型未改/1exact及独立auditPASS/remote='+remote+'；不是确定性field24提高，接受最佳仍11/24+old8。详day_20261113/T513_FINAL_REPORT。Phase隔离。\n\n'+f.read_text())
        subprocess.run(['git','add','stage/docs/day_20261113','stage/docs/long_distance_20261101','stage/docs/RESEARCH_STATUS.md'],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','Report bounded training-only goal-feedback recovery'],cwd=ROOT,check=True)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
        except BaseException:save(DOC/'T513_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_committed=True))
    except BaseException:
        save(DOC/'T513_DRIVER_ERROR.json',dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
