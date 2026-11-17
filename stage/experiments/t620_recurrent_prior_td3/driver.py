"""Only writer for a finite two-arm offline-capable local experiment."""
from pathlib import Path
import os,time,subprocess,signal,traceback,json
import t620_runtime as R
ROOT,HERE,OUT,DOC=R.ROOT,R.HERE,R.OUT,R.DOC
read,save,sha=R.read,R.save,R.sha

def main():
 root=read(OUT/'protocol.json');R.verify(root,True);assert not (OUT/'launch.json').exists()
 receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
 save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],root_tree=receipt['root_tree'],CPU=[14,15],GPU=False,offline_local=True));ledger=dict(state='RUNNING',physics=0,commands=0,main_seconds=0.,children=[],stop_reasons=[]);pids=[];error=None
 def call(action,arm,cap):
  with (OUT/arm/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14,15',str(ROOT/'run_stage.sh'),str(HERE/'t620_runtime.py'),action,arm],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);save(DOC/'STATUS.json',dict(state=action,arm=arm,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=cap);assert c.returncode==0,(action,arm,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 try:
  for arm in root['arms']:
   now=time.time();okay=(now+root['learning_seconds']+30<root['absolute_learning_stop'] and now+root['main_seconds']+30<root['absolute_native_stop'] and now+root['main_seconds']+root['audit_seconds']+root['finalization_seconds']<root['absolute_all_stop'] and ledger['physics']+root['child_max_physics']<=root['max_physics'] and ledger['commands']+root['child_max_commands']<=root['max_commands'] and ledger['main_seconds']+root['main_seconds']<=root['max_main_seconds'] and R.used(OUT)+root['child_storage_reserve']+524288<=root['max_bytes'])
   if not okay:ledger['stop_reasons'].append(dict(arm=arm,reason='FULL_WORST_TIME_BUDGET_STORAGE_ADMISSION_DENIED'));break
   folder=OUT/arm;folder.mkdir();save(folder/'protocol.json',dict(_root_sha256=sha(OUT/'protocol.json'),max_physics=root['child_max_physics'],max_commands=root['child_max_commands'],main_seconds=root['main_seconds'],learning_seconds=root['learning_seconds']))
   try:call('main',arm,root['main_seconds'])
   finally:
    f=folder/'worker_generations.json'
    if f.exists():
     for g in read(f):pids.extend([g['parent']]+g['workers'])
    f=folder/'summary.json'
    if f.exists():
     s=read(f);ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];save(DOC/'LEDGER.json',ledger)
   assert all(not Path('/proc',str(i)).exists() for i in pids);call('audit',arm,root['audit_seconds']);assert read(folder/'audit.json')['state']=='PASS'
   members=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.pyc','.partial']];save(folder/'local_manifest.json',dict(state='SEALED_LOCAL_EVERY_MEMBER_SHA',files=members));assert all(sha(ROOT/z['path'])==z['sha256'] for z in members)
   result=read(folder/'summary.json')['result'];ledger['children'].append(dict(arm=arm,accepted=result['accepted'],check=result['check'],update_counts=dict(actor=result['update']['actor_steps'],critic=result['update']['critic_steps']),audit='PASS'));save(DOC/'LEDGER.json',ledger)
 except BaseException:error=traceback.format_exc();save(DOC/'DRIVER_ERROR.json',dict(error=error,raw_preserved=True,no_training_restart=True,epoch=time.time()))
 gone=all(not Path('/proc',str(i)).exists() for i in pids);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()));ledger['state']='FAILED' if error else 'COMPLETE';save(DOC/'LEDGER.json',ledger);report=dict(state=ledger['state'],error=error,ledger=ledger,all_scientific_processes_stopped=gone,remote='PENDING',full407=None,fresh16=None,reload78=None,globally_promoted=False,epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+root['transport_seconds']+root['source_seconds']+30<root['absolute_all_stop']:
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T620'],cwd=ROOT,capture_output=True,timeout=root['transport_seconds']);assert c.returncode==0,c.stderr.decode(errors='replace')[-1000:];report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T620.json')['state']
  except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
 save(DOC/'FINAL_REPORT.json',report);text='2026-11-17 T620循环prior-TD3对照结束：'+report['state']+'；'+str(ledger['physics'])+'物理/'+str(ledger['commands'])+'命令；逐臂Actor/Critic更新与实际原起点成功见FINAL_REPORT，remote='+report['remote']+'，不称全局/泛化通过。\n\n';f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(text+f.read_text())
 try:
  paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
  for n in ['release_receipt_T620.json','source_manifest_T620.json']:
   if (ROOT/'stage/docs/long_distance_20261101'/n).exists():paths.append('stage/docs/long_distance_20261101/'+n)
  subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report bounded T620 recurrent off-policy pair'],cwd=ROOT,check=True,capture_output=True)
  if time.time()+root['source_seconds']+5<root['absolute_all_stop']:subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=root['source_seconds'])
 except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))

if __name__=='__main__':main()
