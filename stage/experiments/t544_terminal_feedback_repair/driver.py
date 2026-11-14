"""One bounded feedback repair, independent replay audit and complete archive."""
from pathlib import Path
import os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t544_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 p=read(OUT/'protocol.json');R.verify(p,True);end=p['absolute_all_stop'];assert not (OUT/'launch.json').exists();assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<end
 source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');now=time.time()
 save(OUT/'launch.json',dict(pid=os.getpid(),epoch=now,source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),all_stop=end,learning_stop=min(now+p['learning_seconds'],p['absolute_learning_stop']),native_stop=min(now+p['main_seconds'],p['absolute_native_stop']),CPU=[14,15],CPUQuota='100%',GPU=False))
 ids=[]
 def call(action,cap):
  assert time.time()+cap+p['transport_seconds']+p['source_seconds']+10<end
  save(OUT/'driver_status.json',dict(state='RUNNING_'+action,epoch=time.time(),driver_pid=os.getpid()))
  with (OUT/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t544_runtime.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ids.append(c.pid)
   try:c.wait(timeout=cap);assert c.returncode==0,(action,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
  assert not Path('/proc',str(c.pid)).exists()
 try:
  call('main',p['main_seconds']);R.G.stopped(OUT);call('audit',p['audit_seconds']);s=read(OUT/'summary.json');a=read(OUT/'audit.json');assert a['state']=='PASS'
  save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ids,finalizer_writes_only_docs_after_this_point=True,epoch=time.time()));remote='PENDING'
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T544'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['transport_seconds'],end-time.time()-p['source_seconds']-15));assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T544.json')['state']
  except BaseException:save(DOC/'T544_REMOTE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
  report=dict(state='COMPLETE',result=s['result'],audit=a,remote=remote,main_seconds=s['elapsed_s'],main_native=s['physics'],main_commands=s['commands'],engineering_native=p['preflight_physics'],engineering_commands=p['preflight_commands'],new_policy_gradient_updates=0,old275=None,six54=None,eight54=None,fresh16=None,reload78=None,scientific_pids_stopped=ids,epoch=time.time());save(DOC/'T544_FINAL_REPORT.json',report)
  text='2026-11-14 T544实际偏离恢复+MIR启发回放结束，'+s['result']['reason']+'；自主check='+str(s['result']['check'])+'，remote='+remote+'。实际liveActorAdam='+str(s['result']['Actor_Adam'])+'，PG/Value0，shadow/虚拟Adam另计，旧接受T507不覆盖；old275/six54/eight54/fresh/reload未执行null，不称24稳定。详day_20261114/T544_FINAL_REPORT，12:00all/Stage隔离。'
  f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(text+'\n\n'+f.read_text());subprocess.run(['git','add','stage/docs/day_20261114','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report T544 verified feedback repair and raw evidence'],cwd=ROOT,check=True,capture_output=True)
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['source_seconds'],end-time.time()-5));assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:]
  except BaseException:save(DOC/'T544_SOURCE_PENDING.json',dict(error=traceback.format_exc(),raw_preserved=True))
 except BaseException:
  save(DOC/'T544_DRIVER_ERROR.json',dict(error=traceback.format_exc(),known_pids=ids,raw_preserved=True,no_restart=True,epoch=time.time()));raise
if __name__=='__main__':main()
