"""External, audit-boundary-only handoff. Shared night budgets; no LLM/network needed."""
from pathlib import Path
import sys,os,time,signal,subprocess,json,hashlib,traceback,math,shutil,ast
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
DOC=ROOT/'stage/docs/overnight_20261116/t607';PARENT=ROOT/'stage/runs/t606r_offline_retention_recovery';PDOC=ROOT/'stage/docs/overnight_20261116/recovery_001';OUT=ROOT/'stage/runs/t607_taskwise_adam_relay'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def used(p):return sum(x.stat().st_size for x in Path(p).rglob('*') if x.is_file())
def alive(pid):return Path('/proc',str(pid)).exists()
def running(pid):
 try:return Path('/proc',str(pid),'stat').read_text().split(') ')[1].split()[0]!='Z'
 except FileNotFoundError:return False
def active():return subprocess.run(['systemctl','--user','is-active','--quiet','stage-t606r.service']).returncode==0
def identity(pid,needle):return alive(pid) and needle in Path('/proc',str(pid),'cmdline').read_bytes().decode(errors='replace')
def status(state,**kw):save(DOC/'QUEUE_STATUS.json',dict(state=state,pid=os.getpid(),epoch=time.time(),**kw))
def collect():
 summaries=[];pids=[];n_live=n_shadow=n_audit=0
 for folder in sorted(PARENT.iterdir()):
  if not folder.is_dir():continue
  sf=folder/'summary.json';af=folder/'audit.json'
  assert sf.exists() and af.exists(),f'unsealed child {folder.name}'
  s=read(sf);a=read(af);assert s['state']=='COMPLETE' and a['state']=='PASS'
  rows=[]
  for fn in ['cases.jsonl','replays.jsonl']:
   f=folder/fn
   if f.exists():rows.extend(json.loads(x) for x in f.read_text().splitlines() if x)
  assert s['physics']==sum(x['physics'] for x in rows) and s['commands']==(len(rows)+4 if rows else 0)
  for r in rows:assert sha(r['path'])==r['sha256']
  u=(s.get('result') or {}).get('update') or {};steps=u.get('attempted_Adam',0);calls=[]
  f=folder/'optimizer_calls.jsonl'
  if f.exists():calls=[json.loads(x) for x in f.read_text().splitlines() if x]
  for who in ['live','shadow']:
   assert len([x for x in calls if x[0]==who and x[1]=='BEFORE'])==steps
   assert len([x for x in calls if x[0]==who and x[1]=='AFTER'])==steps
  n_live+=steps;n_shadow+=steps;n_audit+=a['historical_discarded_Adam'];pids.append(s['pid'])
  wf=folder/'worker_generations.json'
  if wf.exists():
   for gen in read(wf):pids.extend([gen['parent']]+gen['workers'])
  summaries.append(dict(child=folder.name,physics=s['physics'],commands=s['commands'],main_seconds=s['elapsed_s'],audit='PASS',live=steps,shadow=steps,historical=a['historical_discarded_Adam']))
 assert all(not alive(i) for i in pids)
 return dict(physics=sum(x['physics'] for x in summaries),commands=4+sum(x['commands'] for x in summaries),main_seconds=5+sum(x['main_seconds'] for x in summaries),live=n_live,shadow=n_shadow,historical=n_audit,children=summaries,pids=sorted(set(pids)))

def handoff():
 p=read(PARENT/'protocol.json');paused=None
 status('WAITING_FOR_FROZEN_DIAGNOSTIC_AUDIT_BOUNDARY',latest_authorization='User requests further relay regardless of branch success/failure, same shared caps')
 try:
  while time.time()+2100<p['absolute_learning_stop']:
   if active():
    f=PDOC/'NIGHT_STATUS.json'
    if f.exists():
     s=read(f);child=s.get('child','')
     if s.get('state')=='audit' and child.startswith('stress_') and identity(s['driver_pid'],'t606r_offline_retention_recovery/driver.py') and identity(s['pid'],'t606r_runtime.py'):
      # Main native task already ended; only zero-native independent audit is active.
      assert read(PARENT/child/'summary.json')['state']=='COMPLETE'
      paused=s['driver_pid'];os.kill(paused,signal.SIGSTOP)
      after=read(PDOC/'NIGHT_STATUS.json');children=Path('/proc',str(paused),'task',str(paused),'children').read_text().split()
      if after.get('pid')!=s['pid'] or after.get('state')!='audit' or any(int(x)!=s['pid'] for x in children):
       os.kill(paused,signal.SIGCONT);paused=None;time.sleep(1);continue
      status('PARENT_PAUSED_DURING_ZERO_NATIVE_AUDIT',child=child,parent_pid=paused,audit_pid=s['pid'])
      deadline=time.time()+150
      while running(s['pid']) and time.time()<deadline:time.sleep(.5)
      assert not running(s['pid']) and read(PARENT/child/'audit.json')['state']=='PASS'
      totals=collect();assert totals['physics']<=120000000 and totals['commands']<=4000 and totals['live']==totals['shadow']==totals['historical']<=432
      # Never stop an active learner, simulator, or audit. Frozen parent cannot start another.
      subprocess.run(['systemctl','--user','stop','stage-t606r.service'],check=True,timeout=10);paused=None
      assert not active();save(PARENT/'SEALED.json',dict(all_scientific_processes_stopped=True,scientific_pids=totals['pids'],epoch=time.time(),external_audit_boundary_handoff=True))
      save(DOC/'PARENT_HANDOFF_FINAL_REPORT.json',dict(state='COMPLETE_USER_AUTHORIZED_SECONDARY_DIAGNOSTIC_HANDOFF',totals=totals,original_summaries_unchanged=True,new_learning_under_shared_budget=True,parent_original_finalizer_not_run=True,remote='PENDING',epoch=time.time()))
      members=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(PARENT.rglob('*')) if f.is_file() and f.name!='external_local_manifest.json' and f.suffix not in ['.lock','.partial','.pyc']]
      save(PARENT/'external_local_manifest.json',dict(state='SEALED_LOCAL_EVERY_MEMBER_SHA',files=members));return totals
   else:
    if (PDOC/'FINAL_REPORT.json').exists():
     report=read(PDOC/'FINAL_REPORT.json');assert report['state']=='COMPLETE','Shared scientific evidence contract failed; do not bypass with new learning'
     return collect()
    raise RuntimeError('Parent ended without complete final report; preserve evidence, no new learning')
   time.sleep(1)
  status('NOT_ADMITTED_INSUFFICIENT_COMPLETE_LEARNING_WINDOW');return None
 except BaseException:
  if paused and identity(paused,'t606r_offline_retention_recovery/driver.py'):os.kill(paused,signal.SIGCONT)
  raise

def register(totals):
 assert not active() and not OUT.exists();p=read(PARENT/'protocol.json');original=ROOT/'stage/runs/t606_offline_retention_night'
 own_remaining=60*2**20-used(original)-used(PARENT);remaining=min(432-totals[k] for k in ['live','shadow','historical'])
 if remaining<8 or own_remaining<16*2**20 or 120000000-totals['physics']<2879616 or 4000-totals['commands']<63 or time.time()+2100>=p['absolute_learning_stop']:
  status('NOT_ADMITTED_SHARED_REMAINDER',remaining_Adam=remaining,remaining_bytes=own_remaining,totals=totals);return False
 p.update(id='T607_TASKWISE_ADAM_RELAY',out=str(OUT),arms=['command'],rounds_per_arm=min(4,remaining//8),fit_steps=8,relay_live_Adam_cap=min(32,remaining),max_physics=120000000-totals['physics'],max_commands=4000-totals['commands'],max_main_seconds=36000-math.ceil(totals['main_seconds']),max_bytes=own_remaining,live_Adam_cap=min(32,remaining),shadow_Adam_cap=min(32,remaining),audit_historical_Adam_cap=min(32,remaining),preflight_discarded_Adam_cap=0,transport_seconds=300,parent_transport_seconds=300,reused_native_engineering_audit=str(PARENT/'engineering/audit.json'))
 assert p['max_main_seconds']>=1200 and used(ROOT/'stage/runs')+own_remaining+131072<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 p['shared_parent_totals']=totals;p['shared_parent_totals']['pids']=[]
 for f in HERE.glob('*.py'):ast.parse(f.read_text());p['inputs'][str(f)]=sha(f)
 for f in [DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'METHOD_SOURCES.md',DOC/'PARENT_HANDOFF_FINAL_REPORT.json']:
  if f.exists():p['inputs'][str(f)]=sha(f)
 p['inputs'][p['reused_native_engineering_audit']]=sha(p['reused_native_engineering_audit'])
 for f,h in p['inputs'].items():assert sha(f)==h,f
 OUT.mkdir();blob=json.dumps(p,separators=(',',':'))+'\n';(OUT/'protocol.json').write_text(blob);(HERE/'protocol.json').write_text(blob)
 save(DOC/'SHARED_BUDGET_ADMISSION.json',dict(parent=totals,allocated={k:p[k] for k in ['max_physics','max_commands','max_main_seconds','max_bytes','live_Adam_cap','shadow_Adam_cap','audit_historical_Adam_cap']},epoch=time.time()))
 paths=[str(HERE.relative_to(ROOT)),str(DOC.relative_to(ROOT))]
 subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True,capture_output=True);subprocess.run(['git','commit','-m','Register T607 taskwise Adam relay within shared night budget'],cwd=ROOT,check=True,capture_output=True)
 tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 save(DOC/'LOCAL_SOURCE_REGISTRATION.json',dict(root_tree=tree,local_commit=commit,remote_commit=None,state='LOCAL_COMMIT_LOCKED_REMOTE_PENDING_OFFLINE_AUTHORIZED'))
 status('REGISTERED_AND_DISPATCHING_T607_DRIVER',parent_accounted=True)
 # Original same systemd unit retains CPU/time limits and inhibitor. Driver becomes sole writer.
 os.execv(str(ROOT/'run_stage.sh'),[str(ROOT/'run_stage.sh'),str(HERE/'driver.py')])

def main():
 assert not (DOC/'QUEUE_LAUNCH.json').exists();save(DOC/'QUEUE_LAUNCH.json',dict(pid=os.getpid(),epoch=time.time(),source_hashes={str(f):sha(f) for f in HERE.glob('*.py')},plan_sha256=sha(DOC/'PLAN.md')))
 try:
  total=handoff()
  if total is not None:register(total)
 except BaseException:status('QUEUE_FAILED_NO_TRAINING_RESTART',error=traceback.format_exc())

if __name__=='__main__':main()
