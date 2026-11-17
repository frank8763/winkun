"""Only finalizer for fixed frozen trajectory pair, no recovery/training relay."""
from pathlib import Path
import sys,os,time,subprocess,signal,traceback,json
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t615_original_start_drift_pair';DOC=ROOT/'stage/docs/day_20261117/t615'
read=lambda f:json.loads(Path(f).read_text())
def save(f,j):
 f=Path(f);t=f.with_suffix(f.suffix+'.partial');t.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');os.replace(t,f)
def main():
 p=read(OUT/'protocol.json');assert p['authorization_approved'];assert not (DOC/'DRIVER_LAUNCH.json').exists();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
 save(DOC/'DRIVER_LAUNCH.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],root_tree=receipt['root_tree'],new_Adam=0,new_RL=0));pids=[];error=None
 def call(action,seconds):
  with (OUT/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14,15',str(ROOT/'run_stage.sh'),str(HERE/'t615_runtime.py'),action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);save(DOC/'STATUS.json',dict(state=action,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=seconds);assert c.returncode==0,(action,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 try:
  assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+30<p['absolute_all_stop'];assert time.time()+p['main_seconds']+30<p['absolute_native_stop']
  call('main',p['main_seconds']);workers=read(OUT/'worker_generations.json')
  for w in workers:pids.extend([w['parent']]+w['workers'])
  assert all(not Path('/proc',str(i)).exists() for i in pids);call('audit',p['audit_seconds']);assert read(OUT/'audit.json')['state']=='PASS'
  import hashlib
  members=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['local_manifest.json','driver.log'] and f.suffix not in ['.lock','.pyc','.partial']]
  save(OUT/'local_manifest.json',dict(state='SEALED_LOCAL_EVERY_MEMBER_SHA',files=members));assert all(hashlib.sha256((ROOT/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in members)
 except BaseException:error=traceback.format_exc();save(DOC/'DRIVER_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
 gone=all(not Path('/proc',str(i)).exists() for i in pids);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()));report=dict(state='COMPLETE' if error is None else 'FAILED',error=error,summary=read(OUT/'summary.json') if (OUT/'summary.json').exists() else None,metrics=read(OUT/'METRICS.json') if (OUT/'METRICS.json').exists() else None,audit=read(OUT/'audit.json') if (OUT/'audit.json').exists() else None,all_scientific_processes_stopped=gone,new_Adam=0,new_RL=0,remote='PENDING',epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+p['transport_seconds']+p['source_seconds']+30<p['absolute_all_stop']:
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T615'],cwd=ROOT,capture_output=True,timeout=p['transport_seconds']);assert c.returncode==0,c.stderr.decode(errors='replace')[-1200:];report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T615.json')['state']
  except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
 save(DOC/'FINAL_REPORT.json',report);f=ROOT/'stage/docs/RESEARCH_STATUS.md';text='2026-11-17 T615冻结失守轨迹对照结束：'+report['state']+'；0Adam/0新RL，指标见day_20261117/t615/FINAL_REPORT。原始remote='+report['remote']+'；不称导航改善或通过，T60834/49和field13/24保留，Phase隔离。\n\n';f.write_text(text+f.read_text())
 try:
  paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
  for name in ['source_manifest_T615.json','release_receipt_T615.json']:
   f=ROOT/'stage/docs/long_distance_20261101'/name
   if f.exists():paths.append(str(f.relative_to(ROOT)))
  subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report frozen T615 retention trajectory comparison'],cwd=ROOT,check=True,capture_output=True)
  if time.time()+p['source_seconds']+5<p['absolute_all_stop']:subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=p['source_seconds'])
 except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
if __name__=='__main__':main()
