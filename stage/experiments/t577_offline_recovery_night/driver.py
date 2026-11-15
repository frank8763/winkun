"""Finite local relay: distinct learning arms, native recovery qualification, audits, frozen stress."""
from pathlib import Path
import sys,os,signal,subprocess,time,traceback,copy,math
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t577_runtime as R
ROOT,OUT,DOC,read,save,sha=R.ROOT,R.OUT,R.DOC,R.read,R.save,R.sha

def main():
 root=read(OUT/'protocol.json');R.verify(root,True);assert not (OUT/'launch.json').exists();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert receipt['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),protocol_sha256=sha(OUT/'protocol.json'),source_remote=receipt['remote_commit'],CPU=[14,15],CPUQuota='200%',GPU=False,offline_local_driver=True));ledger=dict(state='RUNNING',physics=0,commands=0,main_seconds=0.,children=[],stop_reasons=[],source_remote=receipt['remote_commit']);save(DOC/'NIGHT_LEDGER.json',ledger);pidlist=[];error=None;arms={a:dict(model=root['static_model'],query=root['static_model'],check=root['initial_check'],corrections=[root['bootstrap_correction']],active=True) for a in root['arms']};cache={};best=None
 def admit(c,learning=False):
  now=time.time();return ledger['physics']+c['max_physics']<=root['max_physics'] and ledger['commands']+c['max_commands']<=root['max_commands'] and ledger['main_seconds']+c['main_seconds']<=root['max_main_seconds'] and now+c['main_seconds']+c['audit_seconds']+30<root['absolute_native_stop'] and now+c['main_seconds']+c['audit_seconds']+180<root['absolute_all_stop'] and (not learning or now+c['learning_seconds']+60<root['absolute_learning_stop']) and R.used(OUT)+c['storage_reserve']+root['final_storage_reserve']<=root['max_bytes']
 def call(action,name,cap):
  folder=OUT/name
  with (folder/(action+'.log')).open('xb') as log:
   c=subprocess.Popen(['taskset','-c','14,15',str(ROOT/'run_stage.sh'),str(HERE/'t577_runtime.py'),action,name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);pidlist.append(c.pid);save(DOC/'NIGHT_STATUS.json',dict(state=action,child=name,pid=c.pid,driver_pid=os.getpid(),epoch=time.time()))
   try:c.wait(timeout=cap);assert c.returncode==0,(action,name,c.returncode)
   except BaseException:
    try:os.killpg(c.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:c.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
    raise
 def child(name,c,learning=False):
  if not admit(c,learning):ledger['stop_reasons'].append(dict(child=name,reason='FULL_WORST_REMAINING_BUDGET_TIME_OR_STORAGE_ADMISSION_DENIED',epoch=time.time()));save(DOC/'NIGHT_LEDGER.json',ledger);return None
  f=OUT/name;f.mkdir();c=dict(c,_root_sha256=sha(OUT/'protocol.json'));save(f/'protocol.json',c)
  try:call('main',name,c['main_seconds'])
  except BaseException:
   if (f/'summary.json').exists():
    ss=read(f/'summary.json');ledger['physics']+=ss['physics'];ledger['commands']+=ss['commands'];ledger['main_seconds']+=ss['elapsed_s'];ledger['children'].append(dict(child=name,state=ss['state'],audit='NOT_COMPLETED',physics=ss['physics'],commands=ss['commands'],partial=True))
   else:ledger['children'].append(dict(child=name,state='NO_FINAL_SUMMARY',physics=None,commands=None,physical_upper_bound=c['max_physics'],command_upper_bound=c['max_commands'],partial=True,raw_preserved=True))
   save(DOC/'NIGHT_LEDGER.json',ledger);raise
  pidlist.extend(R.G.stopped(f));s=read(f/'summary.json');ledger['physics']+=s['physics'];ledger['commands']+=s['commands'];ledger['main_seconds']+=s['elapsed_s'];save(DOC/'NIGHT_LEDGER.json',ledger);call('audit',name,c['audit_seconds']);assert read(f/'audit.json')['state']=='PASS';files=[x for x in f.rglob('*') if x.is_file() and x.name!='local_manifest.json'];m=[dict(path=str(x.relative_to(ROOT)),sha256=sha(x),bytes=x.stat().st_size) for x in files];save(f/'local_manifest.json',dict(state='SEALED_LOCAL_ALL_MEMBER_SHA',files=m));assert all(sha(ROOT/x['path'])==x['sha256'] for x in m);ledger['children'].append(dict(child=name,state=s['state'],physics=s['physics'],commands=s['commands'],elapsed_s=s['elapsed_s'],audit='PASS'));save(DOC/'NIGHT_LEDGER.json',ledger);return s['result']
 try:
  if child('engineering',dict(root['engineering_template'])) is None:raise RuntimeError('ENGINEERING_ADMISSION_FAILED_NO_NAVIGATION_LAUNCH')
  for rnd in range(root['rounds_per_arm']):
   for arm in root['arms']:
    A=arms[arm]
    if not A['active']:continue
    failed=[j for j in root['evaluation_jobs'] if j['id'] not in A['check']['success_ids']];priority=[j for j in failed if j['id'] in root['reference23_success_ids']] or failed
    if not priority:A['active']=False;continue
    target=priority[rnd%len(priority)];key=A['query']['policy_hash']+'|'+target['id'];qualified=False;newdata=False
    if key not in cache:
     c=dict(root['recovery_template'],kind='recovery',arm=arm,initial=A['model'],query_model=A['query'],target_job=target,reference_row=next((r for r in root['reference_rows'] if r['job_id']==target['id']),None));rec=child(f'{arm}_{rnd:02d}_recovery',c)
     if rec is None:A['active']=False;continue
     cache[key]=rec
    rec=cache[key];qualified=rec['teacher_qualified']
    if qualified:
     row=rec['correction'];ident=row['array_digest']
     if ident not in [x['array_digest'] for x in A['corrections']]:A['corrections'].append(row);newdata=True
    if not newdata and arm!='rl':
     ledger['stop_reasons'].append(dict(arm=arm,round=rnd,reason='NO_NEW_QUALIFIED_RECOVERY_DATA_NO_DUPLICATE_FIT'));save(DOC/'NIGHT_LEDGER.json',ledger);continue
    template=root['rl_template'] if arm=='rl' else root['round_template'];c=dict(template,kind='round',arm=arm,initial=A['model'],target_job=target,before_success_ids=A['check']['success_ids'],correction_rows=A['corrections'],training_seeds=[[20261116,root['arms'].index(arm),rnd,k] for k in range(4)])
    result=child(f'{arm}_{rnd:02d}_round',c,True)
    if result is None:A['active']=False;continue
    A['query']=result['proposal'];A['check']=result['check'] if result['accepted'] else A['check']
    if result['accepted']:
     A['model']=result['model'];candidate=dict(arm=arm,model=A['model'],check=A['check']);score=(not candidate['check']['reference23_missing'],len(candidate['check']['success_ids']),-root['arms'].index(arm));oldscore=(-1,-1,-1) if best is None else (not best['check']['reference23_missing'],len(best['check']['success_ids']),-root['arms'].index(best['arm']))
     if score>oldscore:best=candidate
    save(DOC/'ARM_LEDGER.json',dict(arms=arms,best_accepted=best,epoch=time.time()))
   if not any(a['active'] for a in arms.values()):break
  # Final benchmark only for a genuinely accepted reference23-preserving model.
  # Same unchanged baseline407 is cited historically, never repeated as new gain.
  if best is not None and not best['check']['reference23_missing']:
   c=dict(root['final_template'],kind='final',initial=best['model'],arm=best['arm']);child('final407',c)
  else:save(DOC/'FINAL407_NOT_EXECUTED.json',dict(reason='NO_NEW_ACCEPTED_MODEL_PRESERVING_REFERENCE23',historical_T550_not_new_measurement=True))
  # Training is over. Pre-registered frozen paired diagnostics may consume the
  # remaining window; no stress result is used to select a model or train.
  research=best['model'] if best is not None else root['static_model'];models=[root['neural_teacher'],research]
  for idx,jobs in enumerate(root['stress_chunks']):
   cap=max(j['case']['step_cap'] for j in jobs);c=dict(root['stress_template'],kind='stress',arm='stress',initial=research,stress_models=models,stress_jobs=jobs,max_physics=2*sum(j['case']['step_cap'] for j in jobs),max_commands=2*len(jobs)+4)
   if child(f'stress_{idx:03d}',c) is None:break
 except BaseException:error=traceback.format_exc();save(DOC/'NIGHT_ERROR.json',dict(error=error,epoch=time.time(),no_restart=True))
 gone=all(not Path('/proc',str(pid)).exists() for pid in pidlist);save(OUT/'SEALED.json',dict(all_scientific_processes_stopped=gone,scientific_pids=pidlist,epoch=time.time()));ledger['state']='COMPLETE' if error is None else 'FAILED';save(DOC/'NIGHT_LEDGER.json',ledger);report=dict(state=ledger['state'],error=error,ledger=ledger,arms=arms,best_accepted=best,all_scientific_processes_stopped=gone,remote='PENDING',full407=None if not (OUT/'final407/summary.json').exists() else read(OUT/'final407/summary.json')['result'],fresh16=None,reload78=None,globally_promoted=False,epoch=time.time());save(DOC/'FINAL_REPORT.json',report)
 if gone and time.time()+root['transport_seconds']+root['source_seconds']+30<root['absolute_all_stop']:
  try:
   c=subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'T577'],cwd=ROOT,capture_output=True,timeout=root['transport_seconds']);assert c.returncode==0,c.stdout.decode(errors='replace')[-1500:]+c.stderr.decode(errors='replace')[-1500:];report['remote']=read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T577.json')['state']
  except BaseException:save(DOC/'REMOTE_PENDING.json',dict(error=traceback.format_exc(),offline_science_not_blocked=True,epoch=time.time()))
 save(DOC/'FINAL_REPORT.json',report);text='Stage T577通宵已结束。state='+report['state']+'；物理='+str(ledger['physics'])+'，命令='+str(ledger['commands'])+'；保留新模型='+str(best is not None)+'；remote='+report['remote']+'。完整证据与未执行项见FINAL_REPORT.json，不称全局通过。\n';(DOC/'FINAL_REPORT.md').write_text(text);f=ROOT/'stage/docs/RESEARCH_STATUS.md';f.write_text('2026-11-16 '+text+'\n'+f.read_text())
 try:
  subprocess.run(['git','add','stage/docs/overnight_20261115','stage/docs/RESEARCH_STATUS.md','stage/docs/long_distance_20261101'],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Report bounded T577 offline recovery relay'],cwd=ROOT,check=True,capture_output=True)
  if time.time()+root['source_seconds']+5<root['absolute_all_stop']:subprocess.run(['taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'transport.py'),'source_sync'],cwd=ROOT,check=True,capture_output=True,timeout=root['source_seconds'])
 except BaseException:save(DOC/'SOURCE_PENDING.json',dict(error=traceback.format_exc(),epoch=time.time()))
if __name__=='__main__':main()
