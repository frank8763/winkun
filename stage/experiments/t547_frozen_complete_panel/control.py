"""Prepare a reviewable budget; registration requires explicit new allowance."""
from pathlib import Path
import os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t547_runtime as R
ROOT,OUT,DOC,A,read,save,sha=R.ROOT,R.OUT,R.DOC,R.A,R.read,R.save,R.sha

def prepare():
 assert not OUT.exists() and not (HERE/'proposal.json').exists();folder=ROOT/'stage/runs/t546_functional_residual';s=read(folder/'summary.json');assert s['state']=='COMPLETE' and s['result']['panel']['all13_pass'];assert read(folder/'audit.json')['state']=='PASS';R.G.stopped(folder);assert not Path('/proc',str(read(folder/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T546.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=read(folder/'protocol.json');cfg=read(p['actor_configuration']);done={j['id'] for j in p['panel_jobs']};old=[j for j in cfg['old8_jobs'] if j['id'] not in done];field=[j for j in cfg['field_jobs'] if j['id'] not in done];assert len(old)==7 and len(field)==12
 p.update(id='T547_FROZEN_COMPLETE_SAME_POLICY_PANEL',cached_folder=str(folder),initial=s['result']['model'],evaluation_jobs=cfg['old8_jobs']+cfg['field_jobs'],missing_old_jobs=old,missing_field_jobs=field,protected_field_ids=p['initial_check']['field_success_ids'],max_bytes=20*2**20,child_bytes=20*2**20,max_physics=2*sum(j['case']['step_cap'] for j in old+field),max_commands=42,main_seconds=1800,learning_seconds=1,audit_seconds=600,transport_seconds=600,source_seconds=180,scope='Frozen exactly T546 policy hash; no fitting or weight change. Reuse already audited13 only with identical model/jobs; new missing19 and19exact,combine full old8+field24. Full120 for12 new field failures/successes,7oldcompact. No automatic repair or extension; prior successes never traded for new counts. No planner/map/waypoints/runtime teacher/router. Fullold275/six/eight/fresh/reload not arranged, acceptedfalse.',authorization='PENDING_EXPLICIT_ADDITIONAL_DAY_NATIVE_BUDGET;12:00all unchanged')
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [p['actor_configuration'],DOC/'T547_PLAN.md',DOC/'T546_FUNCTIONAL_COMPATIBILITY_SOURCES.md',folder/'protocol.json',folder/'summary.json',folder/'audit.json',folder/'cases.jsonl',folder/'replays.jsonl',p['initial']['path'],ROOT/'stage/docs/long_distance_20261101/release_receipt_T546.json']:p['inputs'][str(f)]=sha(f)
 for row in R.G.rows(folder):p['inputs'][row['path']]=row['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 save(HERE/'proposal.json',p);save(DOC/'T547_PROPOSED_BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',additional_physics=p['max_physics'],additional_commands=42,prior_actual_physics=4928912,prior_actual_commands=240,new_day_worst_physics=4928912+p['max_physics'],new_day_worst_commands=282,original_day_limits=[6500000,240],original_day_limit_increase_requires_confirmation=True,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072,main_seconds=1800,audit_seconds=600,backup_seconds=600,source_seconds=180,all_stop_BJT='2026-11-14 12:00',new_Adam=0,new_fit=0));print('T547_PROPOSAL_ONLY_NOT_REGISTERED',flush=True)

def preflight():
 assert not OUT.exists();p=R.configuration(read(HERE/'proposal.json'));assert time.time()<p['absolute_all_stop'];assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15};R.torch.set_num_threads(1);q=A.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash'];rows=R.cached(p);assert len(rows)==13
 ownjobs={j['id'] for j in p['missing_old_jobs']+p['missing_field_jobs']};cachedjobs={r['job_id'] for r in rows};assert not ownjobs&cachedjobs and len(ownjobs|cachedjobs)==32
 x=R.torch.tensor(R.E.N.S.arrays(rows[0]['path'])['states'][:12]);q.sample=False;q.reset_noise();h=None
 with R.torch.no_grad():
  for i,state in enumerate(x):
   u,l,h,_=q.step(state[None],h);assert R.np.array_equal(u[0].numpy(),R.E.N.S.arrays(rows[0]['path'])['actions'][i])
 save(DOC/'T547_ZERO_NATIVE_READINESS.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),physics=0,commands=0,new_Adam=0,new_fit=0,frozen_model_reload_exact=True,cached13_native_and_replays_same_hash=True,jobs32_disjoint_complete=True,source_registered_worker_reused_no_new_actor=True,initial12commands_exact=True,budget_authorization='PENDING',not_registered=True,not_launched=True));print('T547_ZERO_NATIVE_READY_AWAIT_BUDGET',flush=True)

def register():
 assert not OUT.exists();p=read(HERE/'proposal.json');approval=read(DOC/'T547_BUDGET_AUTHORIZATION.json');assert approval['explicit_user_authorized'] and approval['additional_physics']>=p['max_physics'] and approval['additional_commands']>=42 and approval['all_stop_epoch']==p['absolute_all_stop'];assert read(DOC/'T547_ZERO_NATIVE_READINESS.json')['state']=='PASS_ZERO_NATIVE'
 assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<p['absolute_all_stop'] and time.time()+p['main_seconds']<p['absolute_native_stop'];p['authorization']=approval['user_reply'];p['inputs'][str(DOC/'T547_BUDGET_AUTHORIZATION.json')]=sha(DOC/'T547_BUDGET_AUTHORIZATION.json');OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=read(DOC/'T547_ZERO_NATIVE_READINESS.json')['pid'],reused_verified_same_model_loader_native_contract=True,physics=0,new_Adam=0));R.verify(p,True);print('T547_REGISTERED_NOT_LAUNCHED',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:save(DOC/('T547_'+sys.argv[1]+'_ERROR.json'),dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));raise
