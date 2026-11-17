"""Two matched fresh-policy recollection and noise-schedule arms: main -> all native exits -> independent audit -> archive."""
from pathlib import Path
import os,time,subprocess,signal,traceback,json
import types,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t628r_memory_start_recovery';DOC=ROOT/'stage/docs/day_20261117/t628r'
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);return c
def verify(p):
 used=lambda d:sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
 assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json') and time.time()<p['absolute_all_stop']
 assert used(OUT)<=p['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 for path,h in p['inputs'].items():assert sha(path)==h,path
M=types.SimpleNamespace(ROOT=ROOT,HERE=HERE,OUT=OUT,DOC=DOC,read=read,save=save,verify=verify,setup=lambda p:os.sched_setaffinity(0,set(p['stage_cpus'])),M=types.SimpleNamespace(plan=plan))
def main():
 p=M.M.plan();M.setup(p);M.verify(p);assert not (M.OUT/'launch.json').exists();receipt=M.read(M.ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=M.ROOT,text=True).strip();now=time.time();assert now+p['learning_admission_seconds']<p['absolute_learning_stop'] and now+p['main_seconds']<p['absolute_native_stop'] and now+p['main_seconds']+p['audit_seconds']+p['finalization_seconds']<p['absolute_all_stop'];M.save(M.OUT/'launch.json',dict(pid=os.getpid(),epoch=now,source_remote=receipt['remote_commit'],CPU=p['stage_cpus'],GPU_learning=True,network_required_for_streamed_raw=True));pids=[];error=None
 def call(action,seconds):
  with (M.OUT/(action+'.log')).open('xb') as log:
   c=subprocess.Popen([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t628r_runtime.py'),action],cwd=M.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);M.save(M.DOC/'STATUS.json',dict(state=action,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=seconds);assert c.returncode==0,(action,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 try:
  call('main',p['main_seconds']);g=M.read(M.OUT/'worker_generations.json');pids.extend([i for z in g for i in [z['parent'],*z['workers']]]);assert all(not Path('/proc',str(i)).exists() for i in pids);call('audit',p['audit_seconds']);assert M.read(M.OUT/'audit.json')['state']=='PASS'
 except BaseException:error=traceback.format_exc();M.save(M.DOC/'DRIVER_ERROR.json',dict(error=error,no_restart=True,raw_preserved=True))
 if (M.OUT/'worker_generations.json').exists():pids.extend([i for z in M.read(M.OUT/'worker_generations.json') for i in [z['parent'],*z['workers']]])
 gone=all(not Path('/proc',str(i)).exists() for i in pids);M.save(M.OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()));s=M.read(M.OUT/'summary.json') if (M.OUT/'summary.json').exists() else None;report=dict(state='FAILED' if error else 'COMPLETE',error=error,summary=s,remote='ROOT_PENDING',dataset_remote_batches=len(list(M.OUT.rglob('batch_*_receipt.json'))),no_global_promotion=True,full407=None,fresh16=None,reload78=None,epoch=time.time());M.save(M.DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+p['transport_seconds']+p['source_seconds']+15<p['absolute_all_stop']:
  try:
   subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t628r_transport.py'),'T628R'],cwd=M.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,check=True,timeout=p['transport_seconds']);report['remote']=M.read(M.ROOT/'stage/docs/long_distance_20261101/release_receipt_T628R.json')['state']
  except BaseException:M.save(M.DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),dataset_already_verified_batches=report['dataset_remote_batches']))
 M.save(M.DOC/'FINAL_REPORT.json',report);f=M.ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-17 T628R四任务新策略重采样/固定sigma-vs退火对照结束：'+report['state']+'；实际成功丢失/新增/49与原始remote见t628/FINAL_REPORT，不称泛化或全局晋升。\n\n'+f.read_text())
 try:
  paths=[str(M.DOC.relative_to(M.ROOT)),'stage/docs/RESEARCH_STATUS.md']
  for n in ['release_receipt_T628R.json','source_manifest_T628R.json']:
   if (M.ROOT/'stage/docs/long_distance_20261101'/n).exists():paths.append('stage/docs/long_distance_20261101/'+n)
  subprocess.run(['git','add','--',*paths],cwd=M.ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report fixed-noise versus annealed fresh-policy closed-loop training'],cwd=M.ROOT,check=True,capture_output=True)
  if time.time()+p['source_seconds']+5<p['absolute_all_stop']:subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t628r_transport.py'),'source_sync'],cwd=M.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,capture_output=True,timeout=p['source_seconds'])
 except BaseException:M.save(M.DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
if __name__=='__main__':main()
