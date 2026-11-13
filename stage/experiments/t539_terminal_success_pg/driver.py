"""One bounded native test, audit, raw archive and source finalizer."""
from pathlib import Path
import os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t539_runtime as R
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
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t539_runtime.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ids.append(c.pid)
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
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T539'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['transport_seconds'],end-time.time()-180))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T539.json')['state']
        except BaseException:save(DOC/'T539_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report=dict(state='COMPLETE',result=s['result'],audit=a,remote=remote,main_seconds=s['elapsed_s'],new_native=s['physics'],commands=s['commands'],new_Actor_SGD=1,new_optimizer=1,navigation_test=None,scientific_pids_stopped=ids,epoch=time.time())
        save(DOC/'T539_FINAL_REPORT.json',report)
        f=ROOT/'stage/docs/RESEARCH_STATUS.md'
        f.write_text('2026-11-13 T539整回合安全成功目标一次真实策略梯度结束，'+s['result']['reason']+'，old8='+str(s['result']['old8'])+'，field24='+str(s['result']['field24'])+'；1liveSGD/1shadow/1audit历史重构，0AdamValue/0新增训练采样，旧T519四同policy历史复用明确。原生eval='+str(s['physics'])+'物理，remote='+remote+'。旧最佳独立保持，old275/six54/eight54/fresh/reload未执行；只单Actor部署。详T539_PLAN/FINAL_REPORT，20:00all/Stage隔离。\n\n'+f.read_text())
        subprocess.run(['git','add','stage/docs/day_20261113/T539_FINAL_REPORT.json','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Report compact native state restoration and complete evidence'],cwd=ROOT,check=True,capture_output=True)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(120,end-time.time()-5))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
        except BaseException:save(DOC/'T539_SOURCE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
    except BaseException:
        save(DOC/'T539_DRIVER_ERROR.json',dict(error=traceback.format_exc(),known_pids=ids,raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
