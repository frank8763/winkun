"""Finite local relay. No LLM/network/dependency installation is used for science."""
from pathlib import Path
import sys,os,time,signal,subprocess,traceback,copy
import t613_runtime as R
ROOT,NIGHT,DOC,HERE,read,save,sha=R.ROOT,R.NIGHT,R.DOC,R.HERE,R.read,R.save,R.sha

def main():
 root=read(NIGHT/'protocol.json');R.verify(root,True);assert not (NIGHT/'launch.json').exists()
 receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip()
 save(NIGHT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),source_remote=receipt['remote_commit'],root_tree=receipt['root_tree'],protocol_sha256=sha(NIGHT/'protocol.json'),CPU=[14,15],CPUQuota='200%',GPU=False,offline_local=True))
 cfg=read(root['actor_configuration']);initial=root['student_initial'];initial_ids=root['teacher_initial_success_ids'];ledger=dict(state='RUNNING',physics=0,commands=0,main_seconds=0.,children=[],stop_reasons=[]);save(DOC/'NIGHT_LEDGER.json',ledger)
 arms={a:dict(model=initial,ids=initial_ids,active=True,resume=False,accepted_rounds=0,rejected_rounds=0,consecutive_rejects=0) for a in root['arms']};best=None;pids=[];error=None
 def admit(c,learning=False,tail=True):
  now=time.time();extra=root['final_storage_reserve'] if tail else root['root_tail_bytes']
  okay=(ledger['physics']+c['max_physics']<=root['max_physics'] and ledger['commands']+c['max_commands']<=root['max_commands'] and ledger['main_seconds']+c['main_seconds']<=root['max_main_seconds'] and R.used(NIGHT)+c['storage_reserve']+extra<=root['max_bytes'] and now+c['main_seconds']+30<root['absolute_native_stop'] and now+c['main_seconds']+c['audit_seconds']+root['finalization_seconds']<root['absolute_all_stop'] and (not learning or now+c['learning_seconds']+30<root['absolute_learning_stop']))
  return okay
 def call(action,name,cap):
  f=NIGHT/name
  with (f/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14,15',str(ROOT/'run_stage.sh'),str(HERE/'t613_runtime.py'),action,name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pids.append(c.pid);save(DOC/'NIGHT_STATUS.json',dict(state=action,child=name,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=cap);assert c.returncode==0,(name,action,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 def child(name,c,learning=False,tail=True):
  if not admit(c,learning,tail):ledger['stop_reasons'].append(dict(child=name,reason='FULL_WORST_BUDGET_STORAGE_TIME_ADMISSION_DENIED',epoch=time.time()));save(DOC/'NIGHT_LEDGER.json',ledger);return None
  f=NIGHT/name;f.mkdir();save(f/'protocol.json',dict(c,_root_sha256=sha(NIGHT/'protocol.json'),child_bytes=c['storage_reserve']))
  try:call('main',name,c['main_seconds'])
  except BaseException:
   if (f/'summary.json').exists():
    s=read(f/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];ledger['children'].append(dict(name=name,state=s['state'],physics=s['physics'],commands=s['commands'],audit='NOT_COMPLETED',partial=True))
   else:ledger['children'].append(dict(name=name,state='SUMMARY_MISSING',physics=None,commands=None,upper_physics=c['max_physics'],upper_commands=c['max_commands'],partial=True))
   save(DOC/'NIGHT_LEDGER.json',ledger);raise
  pids.extend(R.G.stopped(f));s=read(f/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];save(DOC/'NIGHT_LEDGER.json',ledger)
  call('audit',name,c['audit_seconds']);assert read(f/'audit.json')['state']=='PASS'
  members=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(f.rglob('*')) if x.is_file() and x.name!='local_manifest.json' and x.suffix not in ['.lock','.pyc']]
  save(f/'local_manifest.json',dict(state='SEALED_LOCAL_EVERY_MEMBER_SHA',files=members));assert all(sha(ROOT/z['path'])==z['sha256'] for z in members)
  ledger['children'].append(dict(name=name,state=s['state'],physics=s['physics'],commands=s['commands'],main_seconds=s['elapsed_s'],audit='PASS'));save(DOC/'NIGHT_LEDGER.json',ledger);return s['result']
 try:
  assert read(root['reused_engineering_audit'])['state']=='PASS'
  assert read(root['reused_matching_data_optimizer_preflight_audit'])['state']=='PASS'
  for arm in root['arms']:
   A=arms[arm]
   if arm=='rl' and best is not None:A.update(model=best['model'],ids=best['ids'],resume=False)
   for rnd in range(root['rounds_per_arm']):
    if not A['active']:break
    failed=[j for j in cfg['evaluation_jobs'] if j['id'] not in A['ids']];priority=[j for j in failed if j['id'] in root['repair_ids']] or failed
    if not priority:break
    c=dict(root['round_template'],kind='round',arm=arm,student_initial=A['model'],resume_optimizer=False,before_success_ids=A['ids'],memory_manifest=root['memory_manifest'],historical_same_policy_training_rows=root['historical_same_policy_training_rows'])
    if arm=='rl':
     c.update(root['rl_template']);c.update(training_job=root['target_jobs'][rnd],training_seeds=[[60820261117,2,rnd,k] for k in range(4)])
    res=child(f'{arm}_{rnd:02d}',c,True)
    if res is None:A['active']=False;break
    if res['accepted']:
     A.update(model=res['model'],ids=res['check']['success_ids'],resume=True,accepted_rounds=A['accepted_rounds']+1,consecutive_rejects=0)
     if len(A['ids'])>=len(initial_ids) and res.get('new_recovery_safe'):
      cand=dict(arm=arm,model=A['model'],ids=A['ids'],check=res['check'],handoff=res['handoff'])
      if best is None or len(cand['ids'])>len(best['ids']):best=cand
    else:
     A['rejected_rounds']+=1;A['consecutive_rejects']+=1
     # Same deterministic BC before-state/data would produce an identical retry.
     if arm!='rl' or not res['update']['applied_parameter_steps'] or A['consecutive_rejects']>=2:A['active']=False
    save(DOC/'ARM_LEDGER.json',dict(arms=arms,best_improved=best,epoch=time.time()))
  if root.get('allow_final407',False) and best is not None and not best['check']['lost_initial']:
   child('final407',dict(root['final_template'],kind='final',arm=best['arm'],student_initial=best['model']),tail=False)
  else:save(DOC/'FINAL407_NOT_EXECUTED.json',dict(reason='FINAL407_NOT_REGISTERED_IN_THIS_MINIMAL_RECOVERY_PILOT',not_zero_percent_failure=True))
  # Locked-model paired stochastic diagnostics. No result affects selection/training.
  research=best['model'] if best is not None else initial;models=[root['base_model'],research]
  for k in range(root['stress_chunks']):
   ids=[(8*k+i)%len(cfg['evaluation_jobs']) for i in range(8)];jobs=[cfg['evaluation_jobs'][i] for i in ids];seeds=[[60620261116,99,8*k+i] for i in range(8)]
   c=dict(root['stress_template'],kind='stress',arm='stress',student_initial=research,stress_models=models,stress_jobs=jobs,stress_seeds=seeds,max_commands=20,max_physics=2*sum(j['case']['step_cap'] for j in jobs))
   if child(f'stress_{k:03d}',c,tail=False) is None:break
 except BaseException:error=traceback.format_exc();save(DOC/'NIGHT_ERROR.json',dict(error=error,epoch=time.time(),no_training_restart=True,partial_raw_preserved=True))
 gone=all(not Path('/proc',str(pid)).exists() for pid in pids);save(NIGHT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pids,epoch=time.time()))
 ledger['state']='COMPLETE' if error is None else 'FAILED';save(DOC/'NIGHT_LEDGER.json',ledger)
 report=dict(state=ledger['state'],error=error,ledger=ledger,arms=arms,best_improved=best,all_scientific_processes_stopped=gone,remote='PENDING',full407=read(NIGHT/'final407/summary.json')['result'] if (NIGHT/'final407/summary.json').exists() else None,fresh16=None,reload78=None,globally_promoted=False,stress_used_for_selection=False,epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+root['transport_seconds']+root['source_seconds']+30<root['absolute_all_stop']:
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T613'],cwd=ROOT,capture_output=True,timeout=root['transport_seconds']);assert c.returncode==0,c.stdout.decode(errors='replace')[-1000:]+c.stderr.decode(errors='replace')[-1000:]
   report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T613.json')['state']
  except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),offline_local_science_completed=True,epoch=time.time()))
 save(DOC/'FINAL_REPORT.json',report)
 text='T613合格恢复后缀监督学习已结束：'+report['state']+'；物理'+str(ledger['physics'])+'、命令'+str(ledger['commands'])+'，新研究成功改进='+str(best is not None)+'；remote='+report['remote']+'。各臂与冻结扰动分列，fresh16/reload78未执行，不称全局/泛化通过。\n'
 (DOC/'FINAL_REPORT.md').write_text(text);status=ROOT/'stage/docs/RESEARCH_STATUS.md';status.write_text('2026-11-17 '+text+'\n'+status.read_text())
 try:
  paths=[str(DOC.relative_to(ROOT)),'stage/docs/RESEARCH_STATUS.md']
  for name in ['source_manifest_T613.json','release_receipt_T613.json']:
   f=ROOT/'stage/docs/long_distance_20261101'/name
   if f.exists():paths.append(str(f.relative_to(ROOT)))
  subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report bounded offline T613 retention relay'],cwd=ROOT,check=True,capture_output=True)
  if time.time()+root['source_seconds']+5<root['absolute_all_stop']:subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=root['source_seconds'])
 except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))

if __name__=='__main__':main()
