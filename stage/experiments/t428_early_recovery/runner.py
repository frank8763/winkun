"""Register, zero-native preflight and unique launch for recovery diagnosis."""
from t428_core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,importlib.util,dis,builtins,types

def prepare():
 assert not OUT.exists();DOC.mkdir(exist_ok=True);prior=ROOT/'stage/runs/t427_fixed_representation';previous=read(prior/'protocol.json');learn=read(prior/'learn/summary.json')['result'];assert read(prior/'learn/audit.json')['state']=='PASS';check=read(learn['check_file']);failed=[r for r in check['rows'] if not r['success']];assert len(failed)==7 and check['wide54']['success']==51 and check['old275']['success']==271
 driver=read(ROOT/'stage/docs/repair_20261108/DRIVER_STATUS.json');assert not Path('/proc',str(driver['driver_pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T427.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=copy.deepcopy(previous);p.update(id='T428_ZERO_TRAINING_ACTUAL_EARLY_DEPARTURE_RECOVERY',out=str(OUT),student_start_model=learn['proposed_model']['path'],student_start_sha256=learn['proposed_model']['sha256'],expected_policy_hash=learn['proposed_model']['policy_hash'],source_model=learn['proposed_model'],failed_rows=failed,student_model=learn['proposed_model']['path'],max_physics=1100000,max_commands=60,max_bytes=10*2**20,main_seconds=900,root_limits=dict(max_physics=1100000,max_commands=100,max_bytes=12*2**20),stage_budgets=dict(native=dict(main_seconds=900,audit_seconds=180,max_physics=1100000,max_commands=60,max_bytes=10*2**20)),preflight_zero_physics_commands=8,preflight_fresh_clone_Adam=0,validation_reserve_seconds=0,backup_seconds=300,source_seconds=120,new_authorization='User2026Nov8 nowresolvefailures; newzeroAdamdiagnosticbeforeexisting20:30native21allstop; no newtrainingafter20:00, no T427restart.',trigger_distance_m=.1,trigger_yaw_deg=5.,trigger_consecutive_holds=3,trigger_backoff_holds=1)
 paths=set(p['inputs'])|{str(f) for f in HERE.glob('*.py')}|{r['path'] for r in failed}|{p['student_start_model'],str(prior/'learn/development.json'),str(prior/'learn/summary.json'),str(prior/'learn/audit.json'),str(ROOT/'stage/docs/long_distance_20261101/release_receipt_T427.json'),str(DOC/'PLAN.md')};p['inputs']={path:sha(path) for path in sorted(paths)}
 used=E.N.K.S.used(ROOT/'stage/runs');assert used+p['root_limits']['max_bytes']<2**30 and shutil.disk_usage(ROOT).free>=40*2**30;assert time.time()+900+180+120+300+120<p['deadline_all'] and time.time()+900<p['deadline_native'];OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',root=p['root_limits'],main=p['stage_budgets'],global_full_reserved=used+p['root_limits']['max_bytes'],native=1100000,new_Actor_Adam=0,all_today_with_full_new_budget=44881494,existing_deadlines_not_extended=True));print('T428_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
 p=read(OUT/'protocol.json');verify(p);assert not (OUT/'preflight.json').exists();ban_learning();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
 for file in HERE.glob('*.py'):
  spec=importlib.util.spec_from_file_location('stage428_check_'+file.stem,file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  def check(code):
   bad={i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL' and i.argval not in vars(module) and not hasattr(builtins,i.argval)}
   for item in code.co_consts:
    if isinstance(item,types.CodeType):bad|=check(item)
   return bad
  for fun in vars(module).values():
   if isinstance(fun,types.FunctionType) and fun.__module__==module.__name__:assert not check(fun.__code__),(file.name,check(fun.__code__))
 folder=OUT/'preflight_native';folder.mkdir();counts=mp.get_context('spawn').RawArray('q',5);pool=Pool(dict(p,out=str(folder),native_stop_epoch=time.time()+180,max_physics=1),counts,lambda:None)
 try:
  ids=[c.pid for c in pool.children];values=pool.request([dict(model=p['student_start_model'],job=p['failed_rows'][0]['request']['job'],mode='actor_teacher',first=0,seed=None,cancel_probe=True,teacher_model=p['teacher_A_model'],teacher_sha256=p['teacher_A_sha256'])]*4);assert all(v['state']=='CANCEL_PASS' for v in values) and sum(counts)==0
 finally:pool.close()
 assert all(not Path('/proc',str(pid)).exists() for pid in ids);save(OUT/'preflight.json',dict(state='PASS',native=0,Adam=0,actualfourworkerscancelledandexit=True,zero_commands=8,inputs_actual_globals=True));E.N.manifest(OUT);print('T428_PREFLIGHT_PASS',flush=True)

def launch():
 p=read(OUT/'protocol.json');verify(p);assert read(OUT/'preflight.json')['state']=='PASS' and not (OUT/'launch.json').exists();assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();receipt=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');tree=subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip();assert receipt['state']=='REMOTE_HEAD_AND_FULL_ROOT_TREE_VERIFIED' and receipt['root_tree']==tree;now=time.time();assert now+900<p['deadline_native'];limit=p['deadline_all']-now-10;assert limit>1800
 with (DOC/'CONSOLE.log').open('xb') as log:child=subprocess.Popen(['systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T428','--why=Bounded zero-training actual recovery diagnosis','timeout','--signal=TERM','--kill-after=2',str(int(limit)),'taskset','-c','0',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 save(OUT/'launch.json',dict(pid=child.pid,epoch=now,deadline=now+limit,source_remote_commit=receipt['remote_commit'],root_tree=tree,protocol_sha256=sha(OUT/'protocol.json'),research_status_sha256=sha(ROOT/'stage/docs/RESEARCH_STATUS.md'),all_science_offline=True,Actor_Adam=0,no_six_or_eight_acceptance=True));print('T428_ACTUALLY_LAUNCHED',child.pid,flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
