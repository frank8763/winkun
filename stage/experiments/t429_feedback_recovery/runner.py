"""One fixed feedback recovery mechanism; no optimizer or training budget."""
from t429_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,importlib.util,dis,builtins,types

def prepare():
 assert not OUT.exists();DOC.mkdir(exist_ok=True);old=read(ROOT/'stage/runs/t428_early_recovery/protocol.json');prior=ROOT/'stage/runs/t428_early_recovery/native';summary=read(prior/'summary.json');assert summary['state']=='COMPLETE' and read(prior/'audit.json')['state']=='PASS';diagnostics=summary['result']['diagnostics'];failed=[d for d in diagnostics if d.get('recovery_row') and not d['recovery_row']['success']];assert len(failed)==5
 state=read(ROOT/'stage/docs/recovery_20261108/DRIVER_STATUS.json');assert not Path('/proc',str(state['driver_pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T428.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=copy.deepcopy(old);entries=[dict(job_id=d['id'],job=d['baseline_row']['request']['job'],first=d['trigger']['first'],student_row=d['student_row'],failed_mixed_row=d['recovery_row'],reference_path=d['baseline_row']['path'],reference_sha256=d['baseline_row']['sha256']) for d in failed];gains=read(ROOT/'stage/runs/t422_feedback_recovery_night/search/summary.json')['result']['best']['gains'];p.update(id='T429_FIXED_FEEDBACK_CURRENT_STATE_RECOVERY_ONLY',out=str(OUT),recoveries=entries,jobs=[e['job'] for e in entries],gains=gains,max_physics=360000,max_commands=20,max_bytes=8*2**20,main_seconds=600,root_limits=dict(max_physics=360000,max_commands=40,max_bytes=10*2**20),stage_budgets=dict(native=dict(main_seconds=600,audit_seconds=180,max_physics=360000,max_commands=20,max_bytes=8*2**20)),backup_seconds=180,source_seconds=120,new_authorization='Currentuserresolvefailed; newfixedfeedbackmechanism on5actualneuralteacherfailedrecoveries, 0Actor/Adam, noadditionalT428orT427budget; 20:30native21allunchanged.')
 paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{str(prior/'summary.json'),str(prior/'audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T428.json'),str(DOC/'PLAN.md'),str(ROOT/'stage/runs/t422_feedback_recovery_night/search/summary.json')}
 for entry in entries:paths|={entry['student_row']['path'],entry['failed_mixed_row']['path'],entry['reference_path']}
 p['inputs']={path:sha(path) for path in sorted(paths)};used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['root_limits']['max_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30;assert sum(x['job']['case']['step_cap'] for x in entries)*2<=360000;assert time.time()+600<p['deadline_native'] and time.time()+600+180+120+180+120<p['deadline_all'];OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',root=p['root_limits'],main=p['stage_budgets'],global_full_reserved=used+p['root_limits']['max_bytes'],exact_max_physics=2*sum(e['job']['case']['step_cap'] for e in entries),prior_today_physics=44512774,day_with_full_new_cap=44872774,new_Actor_Adam=0,deadline_native='20:30',deadline_all='21:00'));print('T429_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
 p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 for file in HERE.glob('*.py'):
  spec=importlib.util.spec_from_file_location('stage429_check_'+file.stem,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  def check(code):
   bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
   for item in code.co_consts:
    if isinstance(item,types.CodeType):bad|=check(item)
   return bad
  for fun in vars(module).values():
   if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(file.name,check(fun.__code__))
 
 for entry in p['recoveries']:
  ref=R.load_reference(entry['reference_path'],entry['reference_sha256']);a=E.N.S.arrays(entry['student_row']['path']);i=entry['first'];assert E.equal(ref['goal'],a['task_goal_xy']);action,_,_=R.feedback(ref,a['poses'][i],float(a['states'][i,56]),i*.2,i,p['gains'],'progress');assert action.shape==(2,) and np.isfinite(action).all() and abs(action).max()<=1
 folder=OUT/'preflight_native';folder.mkdir();counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
 try:
  ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['recoveries'][0]['job'],mode='feedback',first=p['recoveries'][0]['first'],seed=None,cancel_probe=True,reference_path=p['recoveries'][0]['reference_path'],reference_sha256=p['recoveries'][0]['reference_sha256'],gains=p['gains'])]*4);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
 finally:pool.close()
 assert all(not Path('/proc',str(pid)).exists() for pid in ids);save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,actualfourworkerscancelledandexit=True,zero_commands=8,inputs_actual_globals=True));E.N.manifest(OUT);print('T429_PREFLIGHT_PASS',flush=True)

def launch():
 p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree;now=time.time();assert now+600<p['deadline_native'];limit=p['deadline_all']-now-10;assert limit>1500
 with (DOC/'CONSOLE.log').open('xb') as log:child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T429','--why=Bounded zero-training actual recovery diagnosis','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,Actor_Adam=0,no_six_or_eight_acceptance=True));print('T429_ACTUALLY_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
