"""Network/LLM independent finite relay, one finalizer, strict hard deadline."""
from pathlib import Path
import copy,fcntl,json,os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t550_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 root=read(OUT/'protocol.json');R.verify(root,True);assert not (OUT/'launch.json').exists();assert read(OUT/'native_preflight/audit.json')['state']=='PASS';assert not Path('/proc',str(read(OUT/'native_preflight/summary.json')['pid'])).exists();R.G.stopped(OUT/'native_preflight')
 lock=(OUT/'driver.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.time();end=root['absolute_all_stop'];native_end=root['absolute_native_stop'];learn_end=root['absolute_learning_stop'];source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,all_stop=end,learning_stop=learn_end,native_stop=native_end,source_remote=source['remote_commit'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False))
 engineering=read(OUT/'native_preflight/summary.json');ledger=dict(state='RUNNING',physics=engineering['physics'],commands=engineering['commands'],main_seconds=engineering['elapsed_s'],completed=[],scientific_pids=[engineering['pid'],read(OUT/'native_preflight/audit.json')['pid']],fit_proposals=0,retained_count=0,remote='PENDING_OFFLINE',epoch=start);save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,offline_no_network_or_LLM_for_science=True));model=root['initial'];before=root['before_check'];before_rows=root['before_rows'];protected=root['protected_rows'];stopreason='FINITE_MECHANISMS_COMPLETE';final=None
 def status(state,**kw):ledger.update(state=state,epoch=time.time(),**kw);save(OUT/'NIGHT_LEDGER.json',ledger);save(DOC/'T550_LIVE_STATUS.json',dict(state=state,pid=os.getpid(),epoch=time.time(),physics=ledger['physics'],commands=ledger['commands'],fit_proposals=ledger['fit_proposals'],retained_count=ledger['retained_count'],check=before,**kw))
 def run(action,child,cap):
  assert time.time()+cap+15<end;folder=OUT/child
  with (folder/(action+'.log')).open('xb') as log:
   proc=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t550_runtime.py'),action,child],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['scientific_pids'].append(proc.pid);status('RUNNING_'+action,child=child,active_pid=proc.pid)
   try:proc.wait(timeout=cap);assert proc.returncode==0,(action,child,proc.returncode)
   except BaseException:
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  assert not Path('/proc',str(proc.pid)).exists()
 def manifest(folder):
  files=sorted(f for f in folder.rglob('*') if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.pyc']);rows=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in files];save(folder/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=rows,epoch=time.time()));assert all(sha(ROOT/x['path'])==x['sha256'] for x in rows)
 def execute(child,c):
  folder=OUT/child;assert not folder.exists();folder.mkdir();c.update(_root_sha256=sha(OUT/'protocol.json'),initial=model,protected_rows=protected,before_check=before,before_rows=before_rows);save(folder/'protocol.json',c)
  run('main',child,c['main_seconds']);R.G.stopped(folder);s=read(folder/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];assert ledger['physics']<=root['max_physics'] and ledger['commands']<=root['max_commands'] and ledger['main_seconds']<=root['max_main_seconds'];run('audit',child,root['audit_seconds']);assert read(folder/'audit.json')['state']=='PASS';manifest(folder);ledger['completed'].append(child);status('CHILD_AUDITED_LOCAL_MANIFEST_PASS',child=child);return s
 try:
  final_phys=sum(j['case']['step_cap'] for j in root['final_jobs']);final_cmd=411;sum32=sum(j['case']['step_cap'] for j in root['evaluation_jobs'])
  for target_index,job in enumerate(root['failure_jobs']):
   if job['id'] in before['field_success_ids']:continue
   for kind in root['teacher_kinds']:
    worstphys=2*job['case']['step_cap']+2*sum32;worstcmd=74;cap=1800;child='attempt_'+str(target_index).zfill(2)+'_'+kind.lower();allow=ledger['fit_proposals']<root['max_fit_proposals']
    # Complete worst-case final407 is reserved before every new teacher/fit.
    if time.time()+cap+root['audit_seconds']+root['final_reserve_seconds']+root['audit_seconds']+120>native_end or time.time()+cap>learn_end or ledger['physics']+worstphys+final_phys>root['max_physics'] or ledger['commands']+worstcmd+final_cmd>root['max_commands'] or ledger['main_seconds']+cap+root['final_reserve_seconds']>root['max_main_seconds'] or R.used(OUT)+7*2**20+12*2**20>root['max_bytes']:
     stopreason='NEXT_COMPLETE_BRANCH_OR_FINAL_RESERVE_NOT_ADMISSIBLE';break
    c=dict(kind='attempt',teacher_kind=kind,probe_job=job,allow_fit=allow,max_physics=worstphys,max_commands=74,main_seconds=cap,learning_seconds=1200,child_bytes=7*2**20)
    s=execute(child,c);r=s['result'];ledger['fit_proposals']+=int(r['fit_attempted']);assert ledger['fit_proposals']<=8
    if r['retained']:
     model=r['model'];before=r['check'];before_rows=R.G.rows(OUT/child)[1:];newrow=next(x for x in before_rows if x['job_id']==job['id']);assert newrow['full'] and newrow['success'] and not newrow['collision'];protected=protected+[{k:newrow[k] for k in ['job_id','path','sha256','success','collision']}];ledger['retained_count']+=1;status('RETAINED_NEW_LOCAL_SUCCESS',child=child);break
   if stopreason!='FINITE_MECHANISMS_COMPLETE':break
  status('LEARNING_COMPLETE_FINAL_ADMISSION',stop_reason=stopreason)
  if time.time()+root['final_reserve_seconds']+root['audit_seconds']+120<native_end and ledger['physics']+final_phys<=root['max_physics'] and ledger['commands']+final_cmd<=root['max_commands'] and ledger['main_seconds']+root['final_reserve_seconds']<=root['max_main_seconds'] and R.used(OUT)+12*2**20<root['max_bytes']:
   c=dict(kind='final',max_physics=final_phys,max_commands=411,main_seconds=root['final_reserve_seconds'],learning_seconds=0,child_bytes=12*2**20);s=execute('final407',c);final=s['result']
  else:ledger['final_not_executed_reason']='COMPLETE_WORST_FINAL_RESERVE_NOT_AVAILABLE'
  assert all(not Path('/proc',str(i)).exists() for i in ledger['scientific_pids']);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=ledger['scientific_pids'],finalizer_writes_only_docs_after_this_point=True,epoch=time.time()));status('SCIENTIFIC_CHAIN_COMPLETE',stop_reason=stopreason)
  # Remaining shutdown work never creates new native interactions or learning.
  remote='PENDING_OFFLINE'
  try:
   proc=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T550'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(root['transport_seconds'],end-time.time()-root['source_seconds']-30));assert proc.returncode==0,proc.stdout.decode(errors='replace')[-1500:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T550.json')['state']
  except BaseException:save(DOC/'T550_REMOTE_PENDING.json',dict(error=traceback.format_exc(),all_raw_local_with_manifest=True,no_scientific_restart=True))
  report=dict(state='COMPLETE',stop_reason=stopreason,physics=ledger['physics'],commands=ledger['commands'],main_seconds=ledger['main_seconds'],completed=ledger['completed'],fit_proposals=ledger['fit_proposals'],retained_count=ledger['retained_count'],retained_model=model,development32=before,final407=final,new_Adam=0,new_RL=0,fresh16=None,reload78=None,globally_promoted=False,remote=remote,all_scientific_processes_stopped=True,epoch=time.time());save(DOC/'T550_FINAL_REPORT.json',report)
  text='2026-11-14 T550离线下午接力结束；原生'+str(ledger['physics'])+'物理/'+str(ledger['commands'])+'cmd，拟合'+str(ledger['fit_proposals'])+'保留'+str(ledger['retained_count'])+'，old8/field24='+str(before.get('old8'))+'/'+str(before.get('field24'))+'。两种教师仅训练期，0新Adam/RL，有限记忆非泛化。final407以报告实际为准，fresh/reload未执行不称稳定；remote='+remote+'。Phase隔离。'
  f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(text+'\n\n'+f.read_text());subprocess.run(['git','add','stage/docs/day_20261114','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report bounded T550 offline obstacle recovery relay'],cwd=ROOT,check=True,capture_output=True)
  try:
   proc=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(root['source_seconds'],end-time.time()-5));assert proc.returncode==0,proc.stdout.decode(errors='replace')[-1500:]
  except BaseException:save(DOC/'T550_SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_report_complete=True))
 except BaseException:
  status('FAILED_STOP_NO_RESTART');save(DOC/'T550_DRIVER_ERROR.json',dict(error=traceback.format_exc(),ledger=ledger,all_raw_and_models_preserved=True,epoch=time.time()));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
