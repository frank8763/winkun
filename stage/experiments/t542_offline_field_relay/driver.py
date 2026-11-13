"""Finite offline relay, results select the next registered mechanism locally.

Scientific progression never waits for GitHub/network or a chat wake-up.
Whole-child worst-case admission, one native writer, independent audit and
local per-file manifest precede every model handoff.
"""
from pathlib import Path
import copy,json,os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t542_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def manifest(folder):
 files=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(folder.rglob('*')) if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.partial']]
 save(folder/'local_manifest.json',dict(state='LOCAL_ALL_FILES_SHA_VERIFIED_REMOTE_PENDING',files=files,remote_verified=False))
 assert all(sha(ROOT/x['path'])==x['sha256'] for x in files)

def score(check):return (len(check['field_success_ids']),-check['field24']['contact'],-check['distance_sum'])

def admission(p,usage,over,now,bytes_used,final_cap,final=False):
 a,b,c=usage;fm=0 if final else 3300;fa=0 if final else p['audit_seconds']+p['manifest_seconds'];fp=0 if final else final_cap;fc=0 if final else 411
 seconds=over['main_seconds']+p['audit_seconds']+p['manifest_seconds'];reserve=over.get('child_bytes',p['child_bytes'])+(0 if final else 16*2**20)
 checks=dict(physical=a+over['max_physics']+fp<=p['max_physics'],commands=b+over['max_commands']+fc<=p['max_commands'],main_seconds=c+over['main_seconds']+fm<=p['max_main_seconds'],native_time=now+over['main_seconds']+fm+fa+20<p['absolute_native_stop'],all_time=now+seconds+fm+fa+1200<p['absolute_all_stop'],learning_time=over['kind']!='round' or now+p['learning_seconds']+20<p['absolute_learning_stop'],bytes=bytes_used+reserve<=p['max_bytes'])
 return all(checks.values()),checks

def main():
 p=read(OUT/'protocol.json');R.verify(p,True);end=p['absolute_all_stop'];assert not (OUT/'launch.json').exists()
 assert read(OUT/'preflight.json')['state']=='PASS_ZERO_NEW_NATIVE' and read(OUT/'native_preflight/audit.json')['state']=='PASS'
 source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=source['remote_commit'],source_root_tree=source['root_tree'],protocol_sha256=sha(OUT/'protocol.json'),all_stop=end,CPU=[14,15],CPUQuota='100%',GPU=False,offline_science=True))
 ledger=[dict(child='native_preflight',summary=read(OUT/'native_preflight/summary.json'),audit_sha256=sha(OUT/'native_preflight/audit.json'))];ids=[];reason='FIXED_RELAY_COMPLETE';state='FAILED';error=None;best=None;final=None;started=time.time()
 # Original native engineering probe is real physics and counted in this night.
 def totals():return sum(x['summary']['physics'] for x in ledger),sum(x['summary']['commands'] for x in ledger),sum(x['summary']['elapsed_s'] for x in ledger)
 def status(label,**more):
  a,b,c=totals();save(OUT/'status.json',dict(state=label,driver_pid=os.getpid(),epoch=time.time(),physics=a,commands=b,main_seconds=c,completed_children=len(ledger),**more));save(DOC/'NIGHT_LEDGER.json',dict(children=ledger,physics=a,commands=b,main_seconds=c,last_epoch=time.time()))
 def call(child,action,cap):
  assert time.time()+cap+20<end
  with (OUT/child/(action+'.log')).open('xb') as log:
   proc=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t542_runtime.py'),action,child],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ids.append(proc.pid)
   try:proc.wait(timeout=cap);assert proc.returncode==0,(child,action,proc.returncode)
   except BaseException:
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    raise
  assert not Path('/proc',str(proc.pid)).exists()
 def admit(over,final=False):
  reg=read(p['regression_configuration']);base=read(p['actor_configuration']);fcap=sum(j['case']['step_cap'] for j in reg['old275_jobs']+reg['six_jobs']+reg['wide8_jobs']+base['field_jobs'])
  return admission(p,totals(),over,time.time(),R.used(OUT),fcap,final)
 def run(child,over):
  good,why=admit(over,over['kind']=='final')
  if not good:return None,why
  folder=OUT/child;assert not folder.exists();folder.mkdir();save(folder/'protocol.json',dict(over,_root_sha256=sha(OUT/'protocol.json')));status('RUNNING_'+child)
  call(child,'main',over['main_seconds']);call(child,'audit',p['audit_seconds']);s=read(folder/'summary.json');a=read(folder/'audit.json');assert s['state']=='COMPLETE' and a['state']=='PASS';manifest(folder)
  ledger.append(dict(child=child,summary=s,audit_sha256=sha(folder/'audit.json'),manifest_sha256=sha(folder/'local_manifest.json')));status('CHILD_COMPLETE_'+child);return s['result'],None
 try:
  base=read(p['actor_configuration']);jobs=base['old8_jobs']+base['field_jobs'];cap=2*sum(j['case']['step_cap'] for j in jobs)
  q,why=run('qualify_t539',dict(kind='qualify',initial=p['frozen_candidate'],before_check=p['initial_check'],main_seconds=2400,learning_seconds=0,max_physics=cap,max_commands=68))
  if q is None:reason='QUALIFIER_ADMISSION_REJECTED';raise AssertionError(why)
  best=dict(model=q['model'],check=q['check'],origin='qualify_t539') if q['accepted'] else dict(model=p['source_initial'],check=p['initial_check'],origin='T507_LOCKED_BASELINE')
  arms={name:copy.deepcopy(best) for name in p['arms']};save(DOC/'INITIAL_SELECTION.json',dict(qualifier=q,selected=best,selection_rule='retain_all_prior_successIDs_and_improve_count_or_total_remaining_at_equal_count'))
  stop=False
  for rnd in range(p['rounds_per_arm']):
   for ai,arm in enumerate(p['arms']):
    if best['check']['all_field24_pass']:reason='ALL_FIELD24_DEVELOPMENT_PASSED_STOP_LEARNING';stop=True;break
    current=arms[arm];failed=[j for j in base['field_jobs'] if j['id'] not in current['check']['field_success_ids']]
    failed.sort(key=lambda j:(p['target_priority'].index(j['id']) if j['id'] in p['target_priority'] else len(p['target_priority']),j['id']))
    if not failed:continue
    job=failed[rnd%len(failed)];training=[copy.deepcopy(job) for _ in range(4)]
    worst=2*sum(j['case']['step_cap'] for j in jobs)+8*job['case']['step_cap']
    override=dict(kind='round',arm=arm,round=rnd,initial=current['model'],before_check=current['check'],training_jobs=training,training_seeds=[[p['seed'],ai,rnd,i] for i in range(4)],main_seconds=2400,learning_seconds=1800,max_physics=worst,max_commands=76)
    child=arm+'_'+str(rnd).zfill(2);res,why=run(child,override)
    if res is None:reason='COMPLETE_NEXT_ROUND_AND_FINAL_RESERVE_ADMISSION_STOP';save(DOC/'ADMISSION_STOP.json',dict(child=child,checks=why,epoch=time.time(),no_budget_extension=True));stop=True;break
    if res['accepted']:
     arms[arm]=dict(model=res['model'],check=res['check'],origin=child)
     if score(arms[arm]['check'])>score(best['check']):best=copy.deepcopy(arms[arm])
    save(DOC/'CURRENT_SELECTION.json',dict(best=best,arms=arms,latest_child=child,latest_accepted=res['accepted'],heldout_NOT_used_for_selection=True))
   if stop:break
  # A distinct one-pass full under8 + field audit is useful even when field24
  # is incomplete; never relabel its failures as a complete stability gate.
  reg=read(p['regression_configuration']);fj=reg['old275_jobs']+reg['six_jobs']+reg['wide8_jobs']+base['field_jobs'];assert len(fj)==407
  over=dict(kind='final',initial=best['model'],final_jobs=fj,child_bytes=16*2**20,main_seconds=3300,learning_seconds=0,max_physics=sum(j['case']['step_cap'] for j in fj),max_commands=411)
  final,why=run('final_selected407',over)
  if final is None:save(DOC/'FINAL_NOT_EXECUTED.json',dict(admission=why,missing_is_null_not_failure_rate=True,epoch=time.time()))
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc();reason='SHARED_CONTRACT_OR_PROCESS_FAILURE_STOP_CHAIN';save(DOC/'NIGHT_ERROR.json',dict(error=error,scientific_pids=ids,no_restart=True,all_raw_preserved=True,epoch=time.time()))
 finally:
  # Each launched main is waited/killed as a whole group. Inspect only this
  # registered Stage process chain; never other chats or research processes.
  native_ids=list(ids)
  for f in OUT.glob('*/worker_generations.json'):
   for generation in read(f):native_ids.extend([generation['parent'],*generation['workers']])
  assert all(not Path('/proc',str(i)).exists() for i in native_ids)
  # Include a failed child's real physics in accounting, not just completed ones.
  known={x['child'] for x in ledger}
  for folder in sorted(f for f in OUT.iterdir() if f.is_dir()):
   if folder.name not in known and (folder/'summary.json').exists():ledger.append(dict(child=folder.name,summary=read(folder/'summary.json'),audit_sha256=sha(folder/'audit.json') if (folder/'audit.json').exists() else None,partial=True))
   elif folder.name not in known and (folder/'status.json').exists():
    last=read(folder/'status.json');ledger.append(dict(child=folder.name,summary=dict(state='INTERRUPTED_EXACT_COUNTS_UNKNOWN',physics=last.get('physics',0),commands=last.get('commands',0),elapsed_s=0,counts_lower_bound_only=True,protocol_reserved_max=read(folder/'protocol.json')['max_physics']),partial=True))
  a,b,c=totals();counts_exact=not any(x['summary'].get('counts_lower_bound_only') for x in ledger);assert a<=p['max_physics'] and b<=p['max_commands'] and c<=p['max_main_seconds']+30
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),reason=reason,physics=a if counts_exact else None,physics_lower_bound=a,counts_exact=counts_exact,commands=b,elapsed_s=time.time()-started,main_seconds=c,result=dict(best=best,final=final),error=error,children=len(ledger)))
  status('SCIENTIFIC_ENDED_'+state,best=best)
  save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=sorted(set(native_ids)),finalizer_writes_only_docs_after_this_point=True,epoch=time.time()))
  manifest(OUT)
  remote='PENDING_OFFLINE'
  if time.time()+300<end:
   try:
    cp=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T542'],cwd=ROOT,capture_output=True,timeout=min(p['transport_seconds'],end-time.time()-240));assert cp.returncode==0,cp.stdout.decode(errors='replace')[-1200:];remote=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T542.json')['state']
   except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),local_raw_preserved=True,retry_only_after_network_returns=True))
  report=dict(state=state,reason=reason,physics=a if counts_exact else None,physics_lower_bound=a,counts_exact=counts_exact,commands=b,main_seconds=c,elapsed_s=time.time()-started,best=best,final_development=final,new_field24_all_pass=bool(best and best['check']['all_field24_pass']),fresh16=None,reload78=None,stability_claim=False,remote=remote,all_native_processes_stopped=True,deadline=end,ended_epoch=time.time(),error=error)
  save(DOC/'FINAL_REPORT.json',report);save(DOC/'FINAL_REPORT_SHA.json',dict(path=str(DOC/'FINAL_REPORT.json'),sha256=sha(DOC/'FINAL_REPORT.json')))
  from datetime import datetime
  from zoneinfo import ZoneInfo
  date=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d')
  f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text(date+' T542夜间离线接力结束，'+state+'，'+reason+'，物理'+str(a)+'/命令'+str(b)+'。实际候选、field/旧任务与缺测见overnight_20261113/FINAL_REPORT.json。fresh/reload未执行不算失败率，不冒称全场地稳定；remote='+remote+'。Stage/Phase隔离。\n\n'+f.read_text())
  # Never let network restoration block or resurrect ended science.
  if time.time()+120<end:
   try:
    subprocess.run(['git','add','stage/docs/overnight_20261113','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','commit','-m','Report bounded offline field relay and local evidence'],cwd=ROOT,check=True,capture_output=True)
    cp=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,capture_output=True,timeout=min(p['source_seconds'],end-time.time()-10));assert cp.returncode==0
   except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_final_report_complete=True))
 if state!='COMPLETE':raise RuntimeError(reason)
if __name__=='__main__':main()
