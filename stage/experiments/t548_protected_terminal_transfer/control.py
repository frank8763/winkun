"""Prepare one new finite objective; do not extend old windows/budgets."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,io,gzip,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t548_runtime as R
ROOT,OUT,DOC,L,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists() and not (HERE/'proposal.json').exists();previous=ROOT/'stage/runs/t547_frozen_complete_panel';s=read(previous/'summary.json');assert s['state']=='COMPLETE' and read(previous/'audit.json')['state']=='PASS';R.G.stopped(previous);assert not Path('/proc',str(read(previous/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T547.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 review=read(DOC/'T547_SAVED_FAILURE_REVIEW.json');assert review['failed12_identical_parent_arrays']==12 and review['failed12_with_active_rbf']==0
 p=read(previous/'protocol.json');cfg=read(p['actor_configuration']);key='t502_field_sector0_offset2';row=next(r for r in R.G.rows(previous) if r['job_id']==key);a=R.E.N.S.arrays(row['path']);assert row['timeout'] and not row['collision'];d=R.np.linalg.norm(a['poses'][:,:2]-a['task_goal_xy'],axis=1);hits=R.np.flatnonzero(d<=1.);assert len(hits);first=int(hits[0]);assert first>0;job=next(j for j in cfg['field_jobs'] if j['id']==key)
 protect=p['protected_rows']+[R.G.rows(Path(p['cached_folder']))[0]];assert len(protect)==20 and len({r['job_id'] for r in protect})==20
 end=datetime(2026,11,14,13,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
 p.update(id='T548_ONE_TARGET_PROTECTED_TERMINAL_TRANSFER',baseline_row=row,probe_job=job,teacher_first=first,fixed_intervention=dict(rule='FIRST_ACTUAL_STUDENT_DISTANCE_AT_MOST_1M',hold=first,seconds=float(a['native_counts'][:first].sum()*.002),distance_m=float(d[first]),source_full_sha256=row['sha256']),protected20_rows=protect,maximum_new_centers=128,support_fraction=.49,minimum_feature_separation=1e-10,maximum_condition=1e8,max_physics=2*job['case']['step_cap']+2*sum(j['case']['step_cap'] for j in cfg['old8_jobs']+cfg['field_jobs']),max_commands=74,max_bytes=12*2**20,child_bytes=12*2**20,model_bytes=2*2**20,compact_bytes=12*2**10,full_bytes=2**20,main_seconds=2400,learning_seconds=900,audit_seconds=600,transport_seconds=600,source_seconds=180,absolute_all_stop=end,absolute_learning_stop=end-3600,absolute_native_stop=end-1800,authorization='PENDING_NEW_WINDOW_AND_NATIVE_BUDGET',scope='One known timeout overshoot. Same original128.224s clock/goal/collision, first actual d<=1m training feedback goal teacher. Actual safeGOAL+exact required; then separate fixed compact neural readout once+shadow, preserve all originalT546 arithmetic and20success functions. Full pureActor old8+field24+32exact. Strictcontact<=9 unchanged; partialgain not formal promotion. No Adam/PG/Value; no grids/retries or olddirectory restart. Existing allstop12 remains until explicit new13 authorization.')
 prior_native=sum(read(ROOT/'stage/runs'/folder/'summary.json')['physics'] for folder in ['t543_feedback_joint_repair','t544_terminal_feedback_repair','t545_protected_prefix_repair','t546_functional_residual','t547_frozen_complete_panel'])+3000;assert prior_native==5947108
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [p['actor_configuration'],DOC/'T548_PLAN.md',DOC/'T548_METHOD_BOUNDARY.md',DOC/'T547_SAVED_FAILURE_REVIEW.json',previous/'protocol.json',previous/'summary.json',previous/'audit.json',previous/'cases.jsonl',previous/'replays.jsonl',p['initial']['path'],row['path'],ROOT/'stage/docs/long_distance_20261101/release_receipt_T547.json']:p['inputs'][str(f)]=sha(f)
 for r in protect:p['inputs'][r['path']]=r['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30;save(HERE/'proposal.json',p);save(DOC/'T548_PROPOSED_BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',additional_physics=p['max_physics'],additional_commands=74,prior_actual_physics=prior_native,prior_actual_commands=282,new_day_worst_physics=prior_native+p['max_physics'],new_day_worst_commands=356,requested_new_all_stop_BJT='2026-11-14 13:00',existing_authorized_all_stop_BJT='2026-11-14 12:00',requires_explicit_window_and_native_budget=True,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')+p['max_bytes']+131072,new_Adam=0,new_RL=0,new_Value=0,main_live_closed_form=1,main_shadow_closed_form=1,audit_closed_form_rebuild=1,full_teacher=1,full_candidate=1,compact_candidates=31,all_exact=33,initial_panel=dict(old8=8,field24=12,contacts=10,timeouts=2),main_seconds=2400,audit_seconds=600,backup_seconds=600,source_seconds=180));print('T548_PROPOSAL_ONLY_AWAIT_WINDOW_AND_BUDGET',flush=True)

def preflight():
 assert not OUT.exists();p=R.configuration(read(HERE/'proposal.json'));assert time.time()<datetime(2026,11,14,12,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and set(os.sched_getaffinity(0))<={14,15};R.torch.set_num_threads(1)
 q0=L.Old.frozen(p,p['initial']['path']);q=L.Actor(p,len(q0.repair_centers),1);missing,extra=q.load_state_dict(q0.state_dict(),strict=False);assert set(missing)=={'added_centers','added_radii','added_weights'} and not extra
 a=R.E.N.S.arrays(p['baseline_row']['path']);x=R.torch.tensor(a['states'][:12]);assert R.torch.equal(q.sequence(x),q0.sequence(x));future=x.clone();future[6:]+=1.;assert R.torch.equal(q.sequence(x)[:6],q.sequence(future)[:6]);assert not any(v.requires_grad for v in q.parameters())
 q.sample=False;q.reset_noise();h=None
 with R.torch.no_grad():
  for i,state in enumerate(x):
   u,l,h,_=q.step(state[None],h);assert R.np.array_equal(u[0].numpy(),a['actions'][i])
 # Serialized to RAM only; real exported checkpoint reloader is checked before
 # the candidate's physical tests. No temporary model outside stage/runs.
 b=io.BytesIO();R.torch.save(dict(state=q.state_dict(),architecture=L.ARCH),b);saved=R.torch.load(io.BytesIO(gzip.decompress(gzip.compress(b.getvalue(),mtime=0))),map_location='cpu',weights_only=False);clone=L.Actor(p,len(q0.repair_centers),1);clone.load_state_dict(saved['state']);assert clone.phash()==q.phash()
 for k,v in q0.state_dict().items():assert R.torch.equal(q.state_dict()[k],v)
 assert sha(p['initial']['path'])==p['initial']['sha256'];save(DOC/'T548_ZERO_NATIVE_READINESS.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),physics=0,commands=0,new_fit=0,new_Adam=0,initial_frozen_function_exact=True,step_commands_exact=True,old_RBF_arithmetic_kept_separate=True,original_all_parameters_and_buffers_exact=True,RAM_serialization_exact=True,causal_prefix=True,qualified_teacher_NOT_yet_collected=True,full_new_checkpoint_worker_NOT_yet_physically_exercised=True,budget_and_window='PENDING',not_registered=True,not_launched=True));print('T548_ZERO_NATIVE_READY_AWAIT_APPROVAL',flush=True)

def register():
 assert not OUT.exists();p=read(HERE/'proposal.json');approval=read(DOC/'T548_BUDGET_AUTHORIZATION.json');assert approval['explicit_user_authorized'] and approval['additional_physics']>=p['max_physics'] and approval['additional_commands']>=74 and approval['all_stop_epoch']==p['absolute_all_stop'];assert read(DOC/'T548_ZERO_NATIVE_READINESS.json')['state']=='PASS_ZERO_NATIVE';assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<p['absolute_all_stop'] and time.time()+p['main_seconds']<p['absolute_native_stop']
 p['authorization']=approval['user_reply'];p['inputs'][str(DOC/'T548_BUDGET_AUTHORIZATION.json')]=sha(DOC/'T548_BUDGET_AUTHORIZATION.json');OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=read(DOC/'T548_ZERO_NATIVE_READINESS.json')['pid'],source_readiness_sha256=sha(DOC/'T548_ZERO_NATIVE_READINESS.json'),physics=0,commands=0,new_Adam=0));R.verify(p,True);print('T548_REGISTERED_NOT_LAUNCHED',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:save(DOC/('T548_'+sys.argv[1]+'_ERROR.json'),dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));raise
