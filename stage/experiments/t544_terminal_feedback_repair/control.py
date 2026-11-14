"""One fixed terminal-state recovery mechanism, not a teacher/time grid."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t544_runtime as R
ROOT,OUT,DOC,L,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();old=ROOT/'stage/runs/t543_feedback_joint_repair';a=read(old/'audit.json');s=read(old/'summary.json');assert a['state']=='PASS' and s['result']['Actor_Adam']==0 and not s['result']['teacher_qualified'];R.G.stopped(old)
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T543.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=read(old/'protocol.json');end=datetime(2026,11,14,12,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+5400<end
 p.update(id='T544_TERMINAL_FEEDBACK_ACTION_REPAIR',audited_source_folder=str(old),terminal_trigger_m=1.,nominal_speed_m_s=.14,slow_distance_m=.5,minimum_heading_speed_fraction=.25,curvature_distance_floor_m=.2,physical_wheelbase_m=.6,teacher_first=0,model_bytes=2*2**20,full_bytes=2**20,max_bytes=10*2**20,child_bytes=10*2**20,max_commands=74,main_seconds=3600,learning_seconds=1800,audit_seconds=900,transport_seconds=600,source_seconds=180,absolute_learning_stop=end-3600,absolute_native_stop=end-1800,absolute_all_stop=end,preflight_physics=1000,preflight_commands=6,controller_contract='Training-only point-goal curvature2sin(error)/max(d,.2), strict real physical wheelbase.6, nominal.14m/s capped actualmax, slowd.5/minheading.25. First actual student distance<=1m; no map/path/waypoint/reverse/teleport or runtime deployment control. Preserve original clock/history/120 observations. Executed action labels use preimage-feasible boundary hinge,not invented latent mean.',scope='One terminal feedback teacher from actual candidate state; reused20auditedfullhistoriesfromT543,notnewindependentdata. FullteacherGOAL+exact before16Adam+16virtualMIR eachlive/shadow and independent rebuild. Same freshActorAdam/lr3e-5/old.25field.25recovery.5,actual7old12field success locations. Fullautonomousold8+field24+32exact;retain12/old8=8/contact<=9,noinvalidcensor.0newPG/Value,no grid/restart,old275/six54/eight54/fresh16/reload78null.',authorization='User2026-11-14continueanalyzeandexperiment,12:00allconfirmed. Newboundedterminalcontrolmechanism notT543restart. TodaycombinedT543includingprobe+T544includingprobe<=6.5Mphysical/240cmd. CPU14/15aggregateonecore GPUoff global1GiB free40GiB.')
 base=read(p['actor_configuration']);jobs=base['old8_jobs']+base['field_jobs'];p['max_physics']=2*p['probe_job']['case']['step_cap']+2*sum(j['case']['step_cap'] for j in jobs);assert p['max_physics']+1000+s['physics']+2000<=6500000
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [DOC/'T544_PLAN.md',DOC/'T544_TERMINAL_TEACHER_SOURCES.md',p['actor_configuration'],old/'cases.jsonl',old/'audit.json',old/'summary.json',old/'protocol.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T543.json',DOC/'T544_FAILURE_DIAGNOSIS.json']:p['inputs'][str(f)]=sha(f)
 for r in R.G.rows(old)[:20]:p['inputs'][r['path']]=r['sha256']
 sourceprobe=R.G.rows(old/'native_preflight/baseline')[0];p['engineering_baseline_row']=sourceprobe;p['inputs'][sourceprobe['path']]=sourceprobe['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30;OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 baseline=R.E.N.S.arrays(R.G.rows(old)[19]['path']);first,d=R.prefix_rule(p,baseline);assert first<len(baseline['states']) and float(baseline['native_counts'][:first].sum())*.002<40
 storage=dict(full_teacher=2**20,workers_partial=2*2**20,atomic_transient=2**20,model=2*2**20,compact32=32*12*1024,metadata=2*2**20);assert sum(storage.values())+131072<p['max_bytes']
 save(DOC/'T544_BUDGET.json',dict(main_physics=p['max_physics'],engineering_physics=1000,today_cumulative_worst=p['max_physics']+1000+s['physics']+2000,main_commands=74,engineering_commands=6,today_cumulative_commands_worst=74+6+s['commands']+12,live_Adam=16,live_virtual_discardedAdam=16,shadow_Adam=16,shadow_virtualAdam=16,audit_historical_discardedAdam=16,audit_virtualAdam=16,synthetic_preflight_discardedAdam=8,new_PG=0,Value=0,storage=storage,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],global_limit=2**30,derived_intervention=d,main_seconds=3600,audit_seconds=900,backup_seconds=600,source_seconds=180,all_stop_BJT='2026-11-14 12:00'));print('T544_REGISTERED',p['max_physics'],d,flush=True)

def preflight():
 p=R.configuration(read(OUT/'protocol.json'));R.setup();R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=L.state(obj);q=obj[0];r=R.G.rows(Path(p['audited_source_folder']))[19];a0=R.E.N.S.arrays(r['path']);x=R.torch.tensor(a0['states'][:8]);labels=R.np.asarray([[.4,-1.],[1.,.2],[-1.,.3],[.3,.5],[.1,-.5],[.8,.9],[.7,-.8],[.2,.1]],R.np.float32)
 a=dict(states=x.numpy(),teacher_actions=labels,ownership=R.np.r_[R.np.zeros(2,'u1'),R.np.ones(6,'u1')],stochastic=R.np.zeros(8,'u1'))
 refs=[]
 with R.torch.no_grad():target=q.sequence(x).detach().clone()
 for i in range(19):refs.append(dict(id=('old' if i<7 else 't502_field')+str(i).zfill(2),family=int(i>=7),x=x,target=target.clone()))
 cfg=dict(p,fit_steps=2);aa=L.fit(obj,a,cfg,refs,lambda:None,lambda *_:None);bb=L.fit(shadow,a,cfg,refs,lambda:None,lambda *_:None);assert aa==bb and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'discarded_preflight.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash'];Path(model['path']).unlink()
 for o in [obj,shadow]:L.load(o,before);assert digest(L.state(o))==digest(before)
 # Boundary feasible preimages: outward latent saturation must have zero loss.
 loc=R.torch.tensor([[1.2,-1.3],[.5,-.5]],dtype=R.torch.float64,requires_grad=True);y=R.torch.tensor([[1.,-1.],[1.,-1.]],dtype=R.torch.float64);loss=L.P.action_loss(loc,y);assert loss[0].sum()==0 and loss[1].sum()>0;loss.sum().backward();assert not loc.grad[0].any()
 for pose in [[0.,0.,0.],[.2,.1,1.],[.3,-.2,-1.]]:
  u,st,v=R.feedback(p,pose,[1.,0.],[.04,5.,.6,1.,.5]);assert R.np.isfinite(u).all() and R.np.abs(u).max()<=1 and abs(float(u[1])*.5-st)<1e-7 and abs(.2*(.5*(float(u[0])+1))**2-v)<1e-7
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),new_native=0,discarded_synthetic_Adam=4,discarded_synthetic_virtual_Adam=4,executed_action_feasible_set_negative_boundary_checks=True,virtualAdam_and_shadow_and_rollback_exact=True,new_native_recovery_payload_still_requires_test=True));print('T544_ZERO_NATIVE_PREFLIGHT_PASS',flush=True)

def native_probe():
 p=R.configuration(read(OUT/'protocol.json'));R.verify(p,True);assert not (OUT/'native_preflight.json').exists();folder=OUT/'native_preflight';folder.mkdir();R.setup();R.Base.setup(folder);p.update(out=str(folder),teacher_first=2,max_physics=1000,max_commands=6,main_seconds=120,learning_seconds=0);R.E.N.offline();R.E.L.install_stop_handlers();rt=R.Base.Runtime(p);job=copy.deepcopy(p['probe_job']);job['case']['step_cap']=500
 try:
  rt.pool=R.RecoveryPool(dict(p,native_stop_epoch=rt.deadline,phase='student'),rt.counts,rt.guard);vv=rt.request([job],p['initial'],'ENGINEERING_TERMINAL_CONTROLLER_NOT_NAVIGATION',full=True);a,r=vv[0];b=R.E.N.S.arrays(p['engineering_baseline_row']['path']);R.recovery_audit(p,r,a,b);rt.request([job],p['initial'],'EXACT_ENGINEERING_CONTROLLER',full=True,reference=[r])
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  R.fcntl.flock(rt.lock,R.fcntl.LOCK_UN);rt.lock.close()
 assert sum(rt.counts)==1000 and rt.commands==6;save(folder/'summary.json',dict(state='COMPLETE',pid=os.getpid(),physics=1000,commands=6));save(OUT/'native_preflight.json',dict(state='PASS',pid=os.getpid(),physics=1000,commands=6,Actor=0,actual_controller_mapper_and_payload_owner_audit=True,full_arrays_native_exact=True,short_cap_NOT_navigation_success=True));print('T544_NATIVE_PREFLIGHT_PASS',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:
  save(DOC/('T544_'+sys.argv[1]+'_ERROR.json'),dict(error=traceback.format_exc(),native_not_claimed_zero=True,no_restart=True));raise
