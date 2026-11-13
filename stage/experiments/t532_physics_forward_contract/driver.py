"""Single bounded supervisor for the zero-native model admissibility gate."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t532_contract as R
OUT,DOC,read,save,sha=R.OUT,R.DOC,R.read,R.save,R.sha
def main():
    p=read(OUT/'protocol.json');end=p['absolute_all_stop'];assert not (OUT/'launch.json').exists()
    source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json')
    save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),all_stop=end,CPU=[14,15],CPUQuota='100%',GPU=False,new_native=0))
    ids=[]
    def call(action,cap):
        assert time.time()+cap+360<end
        save(OUT/'status.json',dict(state='RUNNING_'+action,epoch=time.time(),driver_pid=os.getpid()))
        with (OUT/(action+'.log')).open('xb') as log:
            c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t532_contract.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ids.append(c.pid)
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
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T532'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(300,end-time.time()-180))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
            remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T532.json')['state']
        except BaseException:save(DOC/'T532_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
        report=dict(state='COMPLETE',result=s['result'],audit=a,remote=remote,elapsed_main_s=s['elapsed_s'],new_native=0,new_optimizer=0,navigation_test=None,scientific_pids_stopped=ids,epoch=time.time())
        save(DOC/'T532_FINAL_REPORT.json',report)
        status=ROOT/'stage/docs/RESEARCH_STATUS.md'
        status.write_text('2026-11-13 T532既有四轨迹物理模型前向合同结束，'+s['result']['reason']+'，0新native/Actor/Value/模型拟合，完整因果prefix动作hidden与四窗预测独立重算PASS；remote='+remote+'。这是未拟合准静态模型的记录动作前向检查，不是自主闭环梯度、原生方向验证或导航提升，Actor不可学习。旧最佳T507/失败原件保留，原生方向与配对RL尚未启动。详day_20261113/T532_PLAN/FINAL_REPORT；17:30all不扩/Stage隔离。\n\n'+status.read_text())
        subprocess.run(['git','add','stage/docs/day_20261113/T532_FINAL_REPORT.json','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True)
        subprocess.run(['git','commit','-m','Record T532 physical-model forward gate and source evidence'],cwd=ROOT,check=True,capture_output=True)
        try:
            c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(180,end-time.time()-5))
            assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
        except BaseException:save(DOC/'T532_SOURCE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
    except BaseException:
        save(DOC/'T532_DRIVER_ERROR.json',dict(error=traceback.format_exc(),known_pids=ids,raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
