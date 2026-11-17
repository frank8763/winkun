"""Finite native data -> audit -> two GPU fits/audits -> full raw backup."""
from pathlib import Path
import os,time,subprocess,signal,traceback,json,shutil,fcntl
import t625_collect as M
L=M.L
def main():
 p=M.plan();M.setup(p);M.verify(p);assert not (M.OUT/'launch.json').exists();receipt=M.read(M.ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=M.ROOT,text=True).strip()
 used=lambda folder:sum(f.stat().st_size for f in folder.rglob('*') if f.is_file());assert used(M.ROOT/'stage/runs')-used(M.OUT)+p['max_bytes']+131072<2**30 and shutil.disk_usage(M.ROOT).free>=40*2**30
 assert time.time()+p['main_seconds']+15<p['absolute_native_stop'];assert time.time()+p['main_seconds']+p['audit_seconds']+2*(p['main_seconds']+p['audit_seconds'])+p['finalization_seconds']<p['absolute_all_stop']
 M.save(M.OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],CPU=p['stage_cpus'],GPU_training=True,Actor_frozen=True));pids=[];children=[];error=None
 def call(script,action,seconds,arm=None):
  folder=M.OUT/('collect' if arm is None else arm);env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if arm is None else '0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
  with (folder/(action+'.log')).open('xb') as log:
   cmd=[str(M.ROOT/'run_stage.sh'),str(M.HERE/script),action]+([] if arm is None else [arm]);c=subprocess.Popen(cmd,cwd=M.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);M.save(M.DOC/'STATUS.json',dict(state=action,arm=arm or 'collect',pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=seconds);assert c.returncode==0,(script,action,arm,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 def manifest(folder):
  files=[dict(path=str(f.relative_to(M.ROOT)),sha256=M.sha(f),bytes=f.stat().st_size) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json'];M.save(folder/'local_manifest.json',dict(files=files));assert all(M.sha(M.ROOT/z['path'])==z['sha256'] for z in files)
 try:
  (M.OUT/'collect').mkdir();call('t625_collect.py','main',p['main_seconds']);g=M.read(M.OUT/'collect/worker_generations.json');pids.extend([i for z in g for i in [z['parent'],*z['workers']]]);assert all(not Path('/proc',str(i)).exists() for i in pids);call('t625_collect.py','audit',p['audit_seconds']);assert M.read(M.OUT/'collect/audit.json')['state']=='PASS';manifest(M.OUT/'collect')
  lock=(M.ROOT/'stage/runs/.stage_gpu.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for arm in ['regression','rank']:
   now=time.time()
   if not (now+p['learning_seconds']+15<p['absolute_learning_stop'] and now+p['main_seconds']+p['audit_seconds']+p['finalization_seconds']<p['absolute_all_stop']):children.append(dict(arm=arm,state='NOT_ADMITTED_FULL_WORST_TIME'));break
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Other GPU compute active, yield'
   assert used(M.OUT)+900000+131072<p['max_bytes'];(M.OUT/arm).mkdir();call('t625_train.py','main',p['main_seconds'],arm);call('t625_train.py','audit',p['audit_seconds'],arm);assert M.read(M.OUT/arm/'audit.json')['state']=='PASS';manifest(M.OUT/arm);s=M.read(M.OUT/arm/'summary.json');children.append(dict(arm=arm,initial=s['initial'],final=s['final'],calibration_gate_pass=s['calibration_gate_pass'],critic_live=p['updates_per_arm'],critic_shadow=p['updates_per_arm'],critic_audit=p['updates_per_arm'],actor_updates=0))
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 except BaseException:error=traceback.format_exc();M.save(M.DOC/'DRIVER_ERROR.json',dict(error=error,no_restart=True))
 if (M.OUT/'collect/worker_generations.json').exists():pids.extend([i for z in M.read(M.OUT/'collect/worker_generations.json') for i in [z['parent'],*z['workers']]])
 gone=all(not Path('/proc',str(i)).exists() for i in pids);M.save(M.OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()));report=dict(state='FAILED' if error else 'COMPLETE',error=error,children=children,collection=M.read(M.OUT/'collect/summary.json') if (M.OUT/'collect/summary.json').exists() else None,remote='PENDING',actor_updates=0,navigation_validation=None,no_automatic_Actor_update=True,epoch=time.time());M.save(M.DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+p['transport_seconds']+p['source_seconds']+15<p['absolute_all_stop']:
  try:
   subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t625_transport.py'),'T625'],cwd=M.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,check=True,timeout=p['transport_seconds']);report['remote']=M.read(M.ROOT/'stage/docs/long_distance_20261101/release_receipt_T625.json')['state']
  except BaseException:M.save(M.DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc()))
 M.save(M.DOC/'FINAL_REPORT.json',report);f=M.ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-17 T625同目标策略续行回报学习对照结束：'+report['state']+'；0Actor，留出排序和回报误差见FINAL_REPORT，不称导航改善/全局晋升，remote='+report['remote']+'。\n\n'+f.read_text())
 try:
  paths=[str(M.DOC.relative_to(M.ROOT)),'stage/docs/RESEARCH_STATUS.md']
  for n in ['release_receipt_T625.json','source_manifest_T625.json']:
   if (M.ROOT/'stage/docs/long_distance_20261101'/n).exists():paths.append('stage/docs/long_distance_20261101/'+n)
  subprocess.run(['git','add','--',*paths],cwd=M.ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report matched target-continuation critic calibration'],cwd=M.ROOT,check=True,capture_output=True)
  if time.time()+p['source_seconds']+5<p['absolute_all_stop']:subprocess.run([str(M.ROOT/'run_stage.sh'),str(M.HERE/'t625_transport.py'),'source_sync'],cwd=M.ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),check=True,capture_output=True,timeout=p['source_seconds'])
 except BaseException:M.save(M.DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc()))
if __name__=='__main__':main()
