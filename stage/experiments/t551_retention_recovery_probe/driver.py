"""One-hour frozen probe, independent audit and sole archive/source finalizer."""
from pathlib import Path
import os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t551_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 p=read(OUT/'protocol.json');R.verify(p,True);end=p['absolute_all_stop'];assert not (OUT/'launch.json').exists();now=time.time();assert now+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<end
 source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');save(OUT/'launch.json',dict(pid=os.getpid(),epoch=now,all_stop=end,native_stop=p['absolute_native_stop'],source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False,new_Adam=0,new_RL=0));ids=[]
 def call(action,cap):
  assert time.time()+cap+p['transport_seconds']+p['source_seconds']+15<end;save(OUT/'driver_status.json',dict(state='RUNNING_'+action,epoch=time.time(),driver_pid=os.getpid()))
  with (OUT/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t551_runtime.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ids.append(c.pid)
   try:c.wait(timeout=cap);assert c.returncode==0,(action,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
  assert not Path('/proc',str(c.pid)).exists()
 try:
  call('main',p['main_seconds']);R.G.stopped(OUT);call('audit',p['audit_seconds']);s=read(OUT/'summary.json');assert read(OUT/'audit.json')['state']=='PASS';save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,finalizer_writes_only_docs_after_this_point=True,epoch=time.time()));remote='PENDING'
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T551'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['transport_seconds'],end-time.time()-p['source_seconds']-15));assert c.returncode==0,c.stdout.decode(errors='replace')[-1500:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T551.json')['state']
  except BaseException:save(DOC/'T551_REMOTE_PENDING.json',dict(error=traceback.format_exc(),all_raw_preserved=True))
  report=dict(state='COMPLETE',result=s['result'],audit=read(OUT/'audit.json'),remote=remote,physics=s['physics'],commands=s['commands'],elapsed_s=s['elapsed_s'],new_optimizer=0,all_scientific_pids_stopped=True,epoch=time.time());save(DOC/'T551_FINAL_REPORT.json',report)
  f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-14 T551固定8任务三模型归因与4单步转向扰动已完成，'+str(s['physics'])+'物理/'+str(s['commands'])+'cmd，独立28全history/28exact审计PASS，0新策略/优化器；实测各模型结局与扰动恢复见day_20261114/T551_FINAL_REPORT。没有跨场地/完整275/留出新验收，不以4扰动称鲁棒性已解决。remote='+remote+'，Phase隔离，不自动启动下一学习。\n\n'+f.read_text());subprocess.run(['git','add','stage/docs/day_20261114','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report T551 fixed-model retention and recovery probes'],cwd=ROOT,check=True,capture_output=True)
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['source_seconds'],end-time.time()-5));assert c.returncode==0,c.stdout.decode(errors='replace')[-1500:]
  except BaseException:save(DOC/'T551_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_report_complete=True))
 except BaseException:save(DOC/'T551_DRIVER_ERROR.json',dict(error=traceback.format_exc(),known_pids=ids,no_restart=True,all_raw_preserved=True,epoch=time.time()));raise
if __name__=='__main__':main()
