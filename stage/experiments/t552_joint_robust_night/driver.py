"""Offline fixed-arm relay; sole writer/finalizer with worst-case admission."""
from pathlib import Path
import fcntl,json,os,signal,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t552_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'launch.json').exists();assert read(OUT/'native_contract/audit.json')['state']=='PASS';R.G.stopped(OUT/'native_contract');lock=(OUT/'driver.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);end=p['absolute_all_stop'];start=time.time();source=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');save(OUT/'launch.json',dict(pid=os.getpid(),epoch=start,all_stop=end,native_stop=p['absolute_native_stop'],learning_stop=p['absolute_learning_stop'],source_remote=source['remote_commit'],root_tree=source['root_tree'],protocol_sha256=sha(OUT/'protocol.json'),CPU=[14,15],CPUQuota='100%',GPU=False));e=read(OUT/'native_contract/summary.json');ledger=dict(state='RUNNING',physics=e['physics'],commands=e['commands'],main_seconds=e['elapsed_s'],completed=['native_contract'],scientific_pids=[e['pid'],read(OUT/'native_contract/audit.json')['pid']],remote='PENDING_OFFLINE',live_Adam=0,shadow_Adam=0,retained=0,epoch=start);arms={a:dict(model=p['initial'],check=p['before_check'],rounds=0,rejections=0,stopped=False) for a in p['arms']};save(OUT/'driver_started.json',dict(pid=os.getpid(),epoch=start,offline_no_network_or_LLM_for_science=True));final=None;stop='FINITE_ARMS_COMPLETE';failure=None
 def status(state,**kw):
  ledger.update(state=state,epoch=time.time(),**kw);save(OUT/'NIGHT_LEDGER.json',ledger);save(DOC/'NIGHT_STATUS.json',dict(driver_pid=os.getpid(),arms=arms,ledger=ledger,epoch=time.time()))
 def manifest(folder):
  files=sorted(f for f in folder.rglob('*') if f.is_file() and f.name!='local_manifest.json' and f.suffix not in ['.lock','.pyc']);items=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in files];save(folder/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=items,epoch=time.time()));assert all(sha(ROOT/x['path'])==x['sha256'] for x in items)
 def run(action,child,cap):
  folder=OUT/child;assert time.time()+cap+15<end
  with (folder/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'t552_runtime.py'),action,child],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);ledger['scientific_pids'].append(c.pid);status('RUNNING_'+action,child=child,active_pid=c.pid)
   try:c.wait(timeout=cap);assert c.returncode==0,(action,child,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
  assert not Path('/proc',str(c.pid)).exists()
 def execute(child,c):
  folder=OUT/child;assert not folder.exists();folder.mkdir();c['_root_sha256']=sha(OUT/'protocol.json');save(folder/'protocol.json',c);run('main',child,c['main_seconds']);ids=R.G.stopped(folder);ledger['scientific_pids'].extend(i for i in ids if i not in ledger['scientific_pids']);s=read(folder/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];assert ledger['physics']<=p['max_physics'] and ledger['commands']<=p['max_commands'] and ledger['main_seconds']<=p['max_main_seconds'];run('audit',child,p['audit_seconds']);assert read(folder/'audit.json')['state']=='PASS';manifest(folder);ledger['completed'].append(child);status('CHILD_AUDIT_AND_MANIFEST_PASS',child=child);return s
 try:
  finalphys=sum(j['case']['step_cap'] for j in p['final_jobs']);roundphys=2*(sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in p['training_jobs']));manifest(OUT/'native_contract')
  for k in range(p['max_rounds_per_arm']):
   for arm in p['arms']:
    a=arms[arm]
    if a['stopped']:continue
    if time.time()+p['main_seconds']+p['audit_seconds']+p['final_reserve_seconds']+p['audit_seconds']+120>p['absolute_native_stop'] or time.time()+p['learning_seconds']>p['absolute_learning_stop'] or ledger['physics']+roundphys+finalphys>p['max_physics'] or ledger['commands']+88+411>p['max_commands'] or ledger['main_seconds']+p['main_seconds']+p['final_reserve_seconds']>p['max_main_seconds'] or R.used(OUT)+p['child_reserve_bytes']+p['final_bytes']>p['max_bytes']:
     stop='NEXT_FULL_ROUND_PLUS_FINAL_RESERVE_NOT_ADMISSIBLE';a['stopped']=True;continue
    child='round_'+str(k).zfill(2)+'_'+arm;c=dict(kind='round',arm=arm,round=k,initial=a['model'],before_check=a['check'],training_seeds=[552000+100*k+i for i in range(4)],max_physics=roundphys,max_commands=88,main_seconds=p['main_seconds'],learning_seconds=p['learning_seconds'],child_bytes=p['child_reserve_bytes']);s=execute(child,c);r=s['result'];ledger['live_Adam']+=r['live_Adam'];ledger['shadow_Adam']+=r['shadow_Adam'];a['rounds']+=1
    if r['accepted']:a.update(model=r['model'],check=r['check'],rejections=0);ledger['retained']+=1
    else:a['rejections']+=1
    if a['rejections']>=p['arm_rejection_stop']:a['stopped']=True
    status('RETAINED_RESEARCH_PROGRESS' if r['accepted'] else 'REJECTED_COMPLETE_ACTOR_ADAM_ROLLBACK',child=child)
  # Strictly predeclared development-only selection, never fresh data.
  ordered=sorted(p['arms'],key=lambda name:(-len(arms[name]['check']['success_ids']),arms[name]['check']['distance_sum'],p['arms'].index(name)));selected=ordered[0];model=arms[selected]['model'];status('LEARNING_COMPLETE_FINAL_ADMISSION',selected_arm=selected,stop_reason=stop)
  if time.time()+p['final_reserve_seconds']+p['audit_seconds']+120<p['absolute_native_stop'] and ledger['physics']+finalphys<=p['max_physics'] and ledger['commands']+411<=p['max_commands'] and ledger['main_seconds']+p['final_reserve_seconds']<=p['max_main_seconds'] and R.used(OUT)+p['final_bytes']<=p['max_bytes']:
   s=execute('final407',dict(kind='final',initial=model,max_physics=finalphys,max_commands=411,main_seconds=p['final_reserve_seconds'],learning_seconds=0,child_bytes=p['final_bytes']));final=s['result']
  else:ledger['final_not_executed_reason']='COMPLETE_FINAL_WORST_RESERVE_UNAVAILABLE'
 except BaseException:
  failure=traceback.format_exc();stop='SCIENTIFIC_OR_ENGINEERING_CONTRACT_STOP_NO_RESTART';save(DOC/'NIGHT_ERROR.json',dict(error=failure,ledger=ledger,all_raw_models_preserved=True,epoch=time.time()))
 finally:
  # Native child launch groups have already been reaped by run(). Never advance
  # after a failed shared evidence contract; report the partial raw honestly.
  gone=all(not Path('/proc',str(i)).exists() for i in ledger['scientific_pids']);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=ledger['scientific_pids'],finalizer_writes_only_docs_after_this_point=True,epoch=time.time()));status('SCIENTIFIC_CHAIN_COMPLETE' if failure is None else 'FAILED_STOPPED_PRESERVE_RAW',stop_reason=stop)
  report=dict(state='COMPLETE' if failure is None else 'FAILED',error=failure,stop_reason=stop,ledger=ledger,arms=arms,final407=final,fresh16=None,reload78=None,cross_site_NOT_tested=True,globally_promoted=False,all_scientific_pids_stopped=gone,remote='PENDING_OFFLINE',epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
  text='# T552 Stage夜间真实结果\n\n'+('有界接力完成。' if failure is None else '证据合同停止；保留全部原始，不重启。')+'\n\n详见 FINAL_REPORT.json：各臂保留/拒绝与最终同模型407分列；未执行fresh16/reload78不是失败率，不称鲁棒性或跨场地已解决。\n';(DOC/'FINAL_REPORT.md').write_text(text)
  # Networking is strictly after all scientific work, never a relay dependency.
  if gone and time.time()+60<end:
   try:
    cap=min(p['transport_seconds'],end-time.time()-p['source_seconds']-20);assert cap>30;c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T552'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=cap);assert c.returncode==0,c.stdout.decode(errors='replace')[-2000:];report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T552.json')['state']
   except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),complete_raw_local=True,epoch=time.time()))
  save(DOC/'FINAL_REPORT.json',report);f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-15 T552三固定臂离线夜间接力已结束，state='+report['state']+'；物理/命令/保留和最终407以overnight_20261114/FINAL_REPORT为准。fresh/reload未执行null，不称泛化/稳定；remote='+report['remote']+'。Stage/Phase隔离，不自动重启。\n\n'+f.read_text())
  try:
   subprocess.run(['git','add','stage/docs/overnight_20261114','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report T552 bounded joint retention night'],cwd=ROOT,check=True,capture_output=True)
   if time.time()+30<end:
    c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=min(p['source_seconds'],end-time.time()-5));assert c.returncode==0,c.stdout.decode(errors='replace')[-1000:]
  except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),local_report_complete=True,epoch=time.time()))
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
