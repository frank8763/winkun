"""Lightweight offline sole writer; no Torch, network or LLM in the relay."""
from pathlib import Path
import os,time,sys,json,hashlib,subprocess,signal,traceback,copy,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t629_offline_night_relay';DOC=ROOT/'stage/docs/overnight_20261117'
def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def save(f,v):
 f=Path(f);f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(v,indent=2)+'\n')
def used(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
def main():
 p=read(OUT/'protocol.json');assert p['authorization_approved'] and sha(OUT/'protocol.json')==sha(HERE/'protocol.json');assert not (OUT/'launch.json').exists();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert read(ROOT/'stage/runs/t628s_actual_memory_recovery/SEALED.json')['all_scientific_processes_stopped'] and read(ROOT/'stage/runs/t628s_actual_memory_recovery/audit.json')['state']=='PASS'
 for f,h in p['inputs'].items():assert sha(f)==h,f
 assert used(ROOT/'stage/runs')-used(OUT)+p['max_bytes']+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert read(ROOT/'stage/docs/day_20261117/t628s/FINAL_REPORT.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),offline_science=True,network_required=False,source_remote=receipt['remote_commit'],protocol_sha256=sha(OUT/'protocol.json')));ledger=dict(state='RUNNING',physics=0,commands=0,main_seconds=0.,Actor_updates=0,children=[],stops=[]);pids=[];best=None;results={};error=None
 def call(action,name,seconds):
  folder=OUT/name
  with (folder/(action+'.log')).open('xb') as log:
   c=subprocess.Popen([str(ROOT/'run_stage.sh'),str(HERE/'t629_runtime.py'),action,name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);save(DOC/'NIGHT_STATUS.json',dict(action=action,child=name,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=seconds);assert c.returncode==0,(action,name,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 def child(name,kind,**kw):
  nonlocal best
  template=dict(p['templates'][kind]);c=dict(p);c.update(template);c.update(kind=kind,student_initial=p['student_initial']);c.update(kw);now=time.time();learning=kind in ['core','broad','recovery','recovery_rl']
  admissible=(ledger['physics']+c['max_physics']<=p['max_physics'] and ledger['commands']+c['max_commands']<=p['max_commands'] and ledger['main_seconds']+c['main_seconds']<=p['max_main_seconds'] and used(OUT)+c['storage_reserve']+p['final_storage_reserve']<=p['max_bytes'] and now+c['main_seconds']<p['absolute_native_stop'] and now+c['main_seconds']+c['audit_seconds']+p['finalization_seconds']<p['absolute_all_stop'] and (not learning or now+c['learning_seconds']<p['absolute_learning_stop']))
  if not admissible:ledger['stops'].append(dict(child=name,reason='COMPLETE_WORST_BUDGET_TIME_STORAGE_ADMISSION_DENIED'));save(DOC/'NIGHT_LEDGER.json',ledger);return None
  folder=OUT/name;folder.mkdir();save(folder/'protocol.json',c)
  try:call('main',name,c['main_seconds'])
  except BaseException:
   if (folder/'summary.json').exists():
    s=read(folder/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s']
   save(DOC/'NIGHT_LEDGER.json',ledger);raise
  s=read(folder/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];g=read(folder/'worker_generations.json');ids=[pid for row in g for pid in [row['parent'],*row['workers']]];assert all(not Path('/proc',str(pid)).exists() for pid in ids);pids.extend(ids);save(DOC/'NIGHT_LEDGER.json',ledger);call('audit',name,c['audit_seconds']);audit=read(folder/'audit.json');assert audit['state']=='PASS';ledger['Actor_updates']+=audit['independent_historical_Actor'];assert ledger['Actor_updates']<=p['Actor_each_cap']
  files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json'];save(folder/'local_manifest.json',dict(state='LOCAL_ALL_MEMBER_SHA',files=files));assert all(sha(ROOT/r['path'])==r['sha256'] for r in files)
  r=s['result'];results[name]=r;ledger['children'].append(dict(name=name,kind=kind,physics=s['physics'],commands=s['commands'],main_seconds=s['elapsed_s'],audit='PASS',Actor_updates=audit['independent_historical_Actor']));save(DOC/'NIGHT_LEDGER.json',ledger)
  if r.get('accepted'):
   ids=r['final49']['success_ids']
   if best is None or len(ids)>len(best['ids']):best=dict(name=name,model=r['model'],ids=ids)
  return r
 try:
  collected=child('collect','collect')
  if collected is None:raise RuntimeError('Full collection admission denied')
  if collected['old_successes'] and collected['new_successes']:
   child('core','core');child('broad','broad')
  else:ledger['stops'].append(dict(reason='FRESH_BATCH_LACKS_OLD_OR_NEW_SUCCESS_NO_PG'))
  candidate=p['recovery_fallback_model'];candidate_rows=p['recovery_fallback_rows']
  for name in ['core','broad']:
   r=results.get(name,{})
   if r.get('final49') and r['final49']['gained_prior'] and r['final49']['lost_prior'] and set(p['new_ids'])<=set(r['final49']['success_ids']):candidate=r['proposal'];candidate_rows=r['evaluation_rows'];break
  lost=sorted(set(p['before_success_ids'])-{r['job_id'] for r in candidate_rows if r['success'] and not r['collision'] and r['field_detail']['field_qualified_success']});old_ids=lost[:2];jobs=[j for j in p['evaluation_jobs'] if j['id'] in old_ids+p['new_ids']]
  if old_ids:
   recovery=child('recovery','recovery',student_initial=candidate,recovery_old_ids=old_ids,recovery_source_jobs=jobs)
   if recovery and recovery.get('accepted'):child('recovery_rl','recovery_rl',student_initial=recovery['model'],before_success_ids=recovery['final49']['success_ids'],actor_steps=2)
  # Fixed-model profiles do not feed training or candidate selection.
  models=[p['reference_model'],p['recovery_fallback_model']]+([] if best is None else [best['model']]);seen=set()
  for index,model in enumerate(models):
   if model['policy_hash'] in seen:continue
   seen.add(model['policy_hash']);child('profile_'+str(index),'profile',student_initial=model)
  for index,jobs in enumerate(p['neighborhood_chunks']):
   models=[p['reference_model']]+([] if best is None else [best['model']])
   for k,model in enumerate(models):
    if child(f'neighborhood_{index:02d}_{k}','profile',student_initial=model,profile_jobs=jobs,exact_all_profile=True,max_physics=2*sum(j['case']['step_cap'] for j in jobs),max_commands=2*len(jobs)+6) is None:break
 except BaseException:error=traceback.format_exc();save(DOC/'NIGHT_ERROR.json',dict(error=error,no_training_restart=True,raw_preserved=True,epoch=time.time()))
 gone=all(not Path('/proc',str(pid)).exists() for pid in pids);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()));ledger['state']='FAILED' if error else 'COMPLETE';save(DOC/'NIGHT_LEDGER.json',ledger);report=dict(state=ledger['state'],error=error,best=best,ledger=ledger,remote='REMOTE_PENDING_OFFLINE_LOCAL_DATA_COMPLETE',all_scientific_processes_stopped=gone,full407_profiles=[name for name in results if name.startswith('profile_')],fresh16=None,reload78=None,no_global_promotion=True,epoch=time.time());save(DOC/'FINAL_REPORT.json',report);save(OUT/'FINAL_REPORT.json',report)
 if gone and time.time()+p['transport_seconds']+p['source_seconds']+60<p['absolute_all_stop']:
  try:
   c=subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'t629_transport.py'),'T629'],cwd=ROOT,capture_output=True,timeout=p['transport_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));assert c.returncode==0;report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T629.json')['state']
  except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),all_local_data_retained=True,offline_science_not_blocked=True))
 save(DOC/'FINAL_REPORT.json',report)
 # Local-only completion is unconditional. Remote export is a separate bounded
 # post-science operation and never a prerequisite for the next science stage.
 f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-18 T629离线接力结束：'+report['state']+'，保留改进='+str(best is not None)+'，完整结果/原始SHA见overnight_20261117/FINAL_REPORT，remote='+report['remote']+'，非全局晋升。\n\n'+f.read_text())
 try:
  paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']+[str(x.relative_to(ROOT)) for pattern in ['release_receipt_T629*.json','source_manifest_T629*.json'] for x in (ROOT/'stage/docs/long_distance_20261101').glob(pattern)]
  subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report bounded offline T629 relay and all failures'],cwd=ROOT,check=True,capture_output=True)
  if time.time()+p['source_seconds']+10<p['absolute_all_stop']:subprocess.run([str(ROOT/'run_stage.sh'),str(HERE/'t629_transport.py'),'source_sync'],cwd=ROOT,capture_output=True,check=True,timeout=p['source_seconds'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
 except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_science_complete=True))
if __name__=='__main__':main()
