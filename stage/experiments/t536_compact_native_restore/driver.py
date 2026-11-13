"""One bounded native test, audit, raw archive and source finalizer."""
from pathlib import Path
import os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t536_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha
def main():
    p=read(OUT/'protocol.json');R.verify(p,True);end=p['absolute_all_stop']
    assert not (OUT/'launch.json').exists() and time.time()+1080<end
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),all_stop=end,CPU=[14,15],CPUQuota='100%',GPU=False,max_native=p['max_physics']))
    ids=[]
    def call(action,cap):
        assert time.time()+cap+360<end
        save(OUT/'status.json',dict(state='RUNNING_'+action,epoch=time.time(),driver_pid=os.getpid()))
        with (OUT/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t536_runtime.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ids.append(c.pid)
            try:c.wait(timeout=cap);assert c.returncode==0,(action,c.returncode)
            except BaseException:
                try:os.killpg(c.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:c.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(c.pid,signal.SIGKILL);c.wait()
                raise
        assert not Path('/proc',str(c.pid)).exists()
    try:
        call('main',p['main_seconds']);call('audit',p['audit_seconds'])
        s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS'
        save(OUT/'status.json',dict(state='SCIENTIFIC_FINISHED',epoch=time.time(),reason=s['result']['reason']))
        save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
        remote='PENDING'
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T536'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['transport_seconds'],end-time.time()-180))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T536.json')['state']
        except BaseException:save(DOC/'T536_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report=dict(state='COMPLETE',result=s['result'],audit=a,remote=remote,main_seconds=s['elapsed_s'],new_native=s['physics'],commands=s['commands'],prior_T534_native=38900,new_optimizer=0,navigation_test=None,scientific_pids_stopped=ids,epoch=time.time())
        save(DOC/'T536_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text('2026-11-13 T536精简快照原生恢复结束，'+s['result']['reason']+'，真实'+str(s['physics'])+'物理/'+str(s['commands'])+'命令/0ActorValue；四窗完整source与磁盘恢复后缀逐位一致、独立原件auditPASS，remote='+remote+'。T534已用38900物理另计，累计'+str(38900+s['physics'])+'；这仅有限窗口恢复资格，不是导航成功或参数方向梯度。原生方向/学习下一项未启动，旧最佳与失败原件保持。详T536_PLAN/FINAL_REPORT；窗口以注册协议为准，Stage/Phase隔离。\n\n'+f.read_text())
        subprocess.run(['git','add','stage/docs/day_20261113/T536_FINAL_REPORT.json','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report compact native state restoration and complete evidence'],cwd=ROOT,check=True,capture_output=True)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(120,end-time.time()-5))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
        except BaseException:save(DOC/'T536_SOURCE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
    except BaseException:
        save(DOC/'T536_DRIVER_ERROR.json',dict(error=traceback.format_exc(),known_pids=ids,raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
