"""Sequential fixed arms; unique native writer, audit, remote backup, source."""
from pathlib import Path
import os,signal,sys,subprocess,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t576_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 root=read(OUT/'protocol.json');R.verify(root,True);assert not (OUT/'launch.json').exists();start=time.time();end=root['absolute_all_stop'];assert start+root['main_seconds']+root['audit_seconds']+root['transport_seconds']+root['source_seconds']+30<end;receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,source_remote=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(OUT/'protocol.json'),absolute_all_stop=end,CPU=[14,15],CPUQuota='200%',GPU=False,max_live_Adam=512,max_shadow_Adam=512,new_RL=0));pids=[];ledger=dict(state='RUNNING',completed=[],physics=0,commands=0,main_seconds=0,source_remote=receipt['remote_commit']);save(DOC/'LEDGER.json',ledger);error=None
 def call(action,arm,cap):
  assert time.time()+cap+10<end
  with (OUT/arm/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14,15',str(ROOT/'run_stage.sh'),str(HERE/'t576_runtime.py'),action,arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);save(DOC/'STATUS.json',dict(state=action,arm=arm,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=cap);assert c.returncode==0,(action,arm,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
  assert not Path('/proc',str(c.pid)).exists()
 try:
  for arm in root['arms']:
   # Same two fixed mechanisms, no acceptance-based parameter/seed grid.
   assert time.time()+root['per_arm_main_seconds']+root['per_arm_audit_seconds']+root['transport_seconds']+root['source_seconds']+30<end
   call('main',arm,root['per_arm_main_seconds']);pids+=R.G.stopped(OUT/arm);call('audit',arm,root['per_arm_audit_seconds']);assert read(OUT/arm/'audit.json')['state']=='PASS';s=read(OUT/arm/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];ledger['completed'].append(arm);assert ledger['physics']<=root['max_physics'] and ledger['commands']<=root['max_commands'];save(DOC/'LEDGER.json',ledger)
   files=sorted(f for f in (OUT/arm).rglob('*') if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.pyc']);members=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f),bytes=f.stat().st_size) for f in files];save(OUT/arm/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=members));assert all(sha(ROOT/m['path'])==m['sha256'] for m in members)
 except BaseException:error=traceback.format_exc();save(DOC/'DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
 gone=all(not Path('/proc',str(i)).exists() for i in pids);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,finalizer_writes_only_docs_after_this_point=True,epoch=time.time()));ledger['state']='COMPLETE' if error is None else 'FAILED';save(DOC/'LEDGER.json',ledger);report=dict(state=ledger['state'],error=error,ledger=ledger,arms={arm:read(OUT/arm/'summary.json') if (OUT/arm/'summary.json').exists() else None for arm in root['arms']},all_scientific_processes_stopped=gone,remote='PENDING',max_live_Adam=512,max_shadow_Adam=512,new_RL=0,globally_promoted=False,epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+root['transport_seconds']+root['source_seconds']+15<end:
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T576'],cwd=ROOT,capture_output=True,timeout=root['transport_seconds']);assert c.returncode==0,c.stdout.decode(errors='replace')[-1200:]+c.stderr.decode(errors='replace')[-1200:];report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T576.json')['state']
  except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
 save(DOC/'FINAL_REPORT.json',report);p=ROOT/'stage/docs/RESEARCH_STATUS.md';p.write_text('2026-11-15 T576同数据同Adam线性与非线性可塑性对照结束，'+report['state']+'，实际逐臂结果见day_20261115/t576/FINAL_REPORT，remote='+report['remote']+'。每臂最多256live256shadow及独立历史Adam重构/0新RL/无全场地或泛化晋升声明，不自动续预算。\n\n'+p.read_text())
 try:
  subprocess.run(['git','add','stage/docs/day_20261115/t576','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report T576 fixed functional constraint comparison'],cwd=ROOT,check=True,capture_output=True)
  if time.time()+root['source_seconds']+5<end:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,capture_output=True,timeout=root['source_seconds']);assert c.returncode==0
 except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
if __name__=='__main__':main()
