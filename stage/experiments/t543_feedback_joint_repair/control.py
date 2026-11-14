"""Lock one candidate, one divergence rule, one teacher, one repair proposal."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,os,sys,time
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t543_runtime as R
ROOT,OUT,DOC,L,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();end=datetime(2026,11,14,12,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+7300<end
 prior=ROOT/'stage/runs/t542_offline_field_relay';initial=read(prior/'rtg_06/summary.json')['result']['model'];check=read(prior/'rtg_06/summary.json')['result']['check'];assert check['old8']['success']==7 and check['field24']['success']==12
 assert read(prior/'rtg_06/audit.json')['state']=='PASS' and read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T542.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
 actorfile=ROOT/'stage/runs/t531_joint_success_distillation/protocol.json';base=read(actorfile);alljobs=base['old8_jobs']+base['field_jobs'];lookup={r['job_id']:r for r in R.G.rows(prior/'rtg_06') if r['role']=='PURE_STUDENT_FULL32'}
 successful=[j for j in alljobs if lookup[j['id']]['success'] and not lookup[j['id']]['collision'] and lookup[j['id']]['field_detail']['field_qualified_success']];assert len(successful)==19
 oldjob=next(j for j in alljobs if j['id']=='dev_task6_13');jobs=successful+[oldjob];assert len(jobs)==20 and len({j['id'] for j in jobs})==20
 oldrow=next(r for r in R.G.rows(ROOT/'stage/runs/t507_joint_success_transfer/pilot') if r['job_id']==oldjob['id']);assert oldrow['success']
 cap=2*sum(j['case']['step_cap'] for j in jobs)+2*oldjob['case']['step_cap']+2*sum(j['case']['step_cap'] for j in alljobs);assert cap+2000<=6500000
 p=dict(id='T543_STUDENT_FEEDBACK_MIR_JOINT_REPAIR',actor_configuration=str(actorfile),source_protocol=base['source_protocol'],initial=initial,neural_teacher=base['neural_teacher'],source_cases=str(prior/'rtg_06/cases.jsonl'),source_jobs=jobs,probe_job=oldjob,old_success_row=oldrow,initial_check=check,accepted_baseline_field_ids=base['baseline_field_success_ids'],max_accepted_field_contacts=9,teacher_first=0,divergence_m=.03,divergence_deg=3.,fit_steps=16,repair_lr=3e-5,raw_sigma=base['raw_sigma'],model_bytes=2*2**20,full_bytes=int(1.5*2**20),compact_bytes=12*1024,max_bytes=44*2**20,child_bytes=44*2**20,max_physics=cap,max_commands=118,main_seconds=5400,learning_seconds=3600,audit_seconds=900,transport_seconds=600,source_seconds=180,absolute_learning_stop=end-3600,absolute_native_stop=end-1800,absolute_all_stop=end,stage_cpus=[14,15],GPU=False,preflight_physics=2000,preflight_commands=12,scope='One fixed rtg06 rejected12/24 research candidate.20same-policy raw reconstructions(19success+oldleftfailure)and20exact; earliest3cm/3deg vs verifiedT507oldleft at previoushold selects one teacher continuation. Full successful teacher only supplies unclipped neural location labels.16Adam+16virtualMIRsteps perlive/shadow; virtual wholeActorAdam rollback exact; select most harmed2old+3fieldfullhistories,not zero-init memory,weights recovery.5/old.25/field.25. Final singleActorold8+field24 and32exact. Preserve all12+old8=8/contact<=9 gate. No new RL,0Value,no teacher runtime/no map/waypoint/router,standardDAgger/MIRinspired notoriginal or safety certificate. No epoch/teacher/time grid,no restart. Fullold275/six54/eight54/fresh/reload not included.',authorization='2026-11-14 user explicitlycontinue field24 and confirmednew12:00allwindow. New bounded intervention mechanism,not extension ofoldnight. Native main worst'+str(cap)+' plus2000engineeringpreflight;0newRL/0Value.44MiBown/global1GiB/free40GiB CPU14/15aggregateonecore/GPUoff.')
 p['protected_files']={initial['path']:initial['sha256'],p['neural_teacher']['path']:p['neural_teacher']['sha256']}
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [actorfile,DOC/'T543_PLAN.md',DOC/'RECOVERY_PRIMARY_SOURCES.md',DOC/'T543_COLD_MODELS.json',prior/'rtg_06/summary.json',prior/'rtg_06/audit.json',Path(p['source_cases']),Path(oldrow['path'])]:p['inputs'][str(f)]=sha(f)
 for j in jobs:
  r=lookup[j['id']];p['inputs'][r['path']]=r['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<=2**30
 OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 # Known packed full120 state/history/64hidden schema, <=900 holds. This is
 # an uncompressed shape bound; raw payload gzip cannot exceed full1.5MiB.
 a=R.E.N.S.arrays(oldrow['path']);packed=R.E.N.S.packed(a);n=len(a['states']);upper=sum(((900+(v.shape[0]==n+1))*int(v.nbytes/v.shape[0]) if v.ndim and v.shape[0] in [n,n+1] else v.nbytes) for v in packed.values())+900*2*(2*8)+65536;assert upper<p['full_bytes']
 storage=dict(full21=21*p['full_bytes'],worker_partial2=2*p['full_bytes'],model=p['model_bytes'],atomic_full_transient=p['full_bytes'],compact32=32*p['compact_bytes'],metadata=3*2**20);assert sum(storage.values())+131072<p['max_bytes']
 save(DOC/'T543_BUDGET.json',dict(max_main_physics=cap,preflight_physics=2000,combined_physics=cap+2000,main_commands=118,preflight_commands=12,main_seconds=5400,audit_seconds=900,backup_seconds=600,source_seconds=180,own_bytes=p['max_bytes'],global_reserved=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],global_limit=2**30,full_worst_uncompressed_bound=upper,storage=storage,live_ActorAdam=16,live_virtual_discardedAdam=16,shadow_Adam=16,shadow_virtual_discardedAdam=16,independent_rebuild_Adam=16,independent_rebuild_virtual_discardedAdam=16,preflight_synthetic_real_and_virtual_discarded_Adam=8,new_RL=0,Value=0,all_stop_BJT='2026-11-14 12:00',GPU=False,CPU_total_one_core=True));print('T543_REGISTERED_NOT_LAUNCHED',cap,upper,flush=True)

def preflight():
 root=read(OUT/'protocol.json');p=R.configuration(root);R.setup();R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=L.state(obj);q=obj[0];row=p['old_success_row'];a0=R.E.N.S.arrays(row['path']);x=R.torch.tensor(a0['states'][:8]);h=None;teacher=L.A.frozen(p,p['neural_teacher']['path']);teacher.sample=False;tt=[];tl=[]
 with R.torch.no_grad():
  for z in x:
   u,loc,h,_=teacher.step(z[None],h);tt.append(u[0].numpy().copy());tl.append(loc[0].numpy().copy())
 a=dict(states=x.numpy(),ownership=R.np.r_[R.np.zeros(2,'u1'),R.np.ones(6,'u1')],stochastic=R.np.zeros(8,'u1'),teacher_actions=R.np.asarray(tt),teacher_locations=R.np.asarray(tl))
 refs=[]
 with R.torch.no_grad():target=q.sequence(x).detach().clone()
 for i in range(19):refs.append(dict(id=('old' if i<7 else 't502_field')+str(i).zfill(2),family=int(i>=7),x=x,target=target.clone()))
 cfg=dict(p,fit_steps=2);aa=L.fit(obj,a,cfg,refs,lambda:None,lambda *_:None);bb=L.fit(shadow,a,cfg,refs,lambda:None,lambda *_:None);assert aa==bb and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'discarded_preflight.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash'];Path(model['path']).unlink()
 for o in [obj,shadow]:L.load(o,before);assert digest(L.state(o))==digest(before)
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),new_native=0,discarded_synthetic_Adam=4,discarded_synthetic_virtual_Adam=4,shadow_MIR_ranking_and_wholeActorAdam_rollback_exact=True,teacher_locations_not_clipped_labels=True,current_output_route=str(OUT),real_native_recovery_payload_still_requires_test=True));print('T543_ZERO_NATIVE_PREFLIGHT_PASS',flush=True)

def native_probe():
 root=read(OUT/'protocol.json');R.verify(root,True);assert not (OUT/'native_preflight.json').exists();R.E.N.offline();R.E.L.install_stop_handlers();R.setup();results=[];base_arrays=None
 for name,kind in [('baseline',R.F.Pool),('teacher',R.RecoveryPool)]:
  folder=OUT/'native_preflight'/name;folder.mkdir(parents=True);p=R.configuration(root);p.update(out=str(folder),teacher_first=2,max_physics=1000,max_commands=6,main_seconds=120,learning_seconds=0);R.Base.setup(folder);rt=R.Base.Runtime(p);job=copy.deepcopy(p['probe_job']);job['case']['step_cap']=500
  try:
   rt.pool=kind(dict(p,native_stop_epoch=rt.deadline,phase='student'),rt.counts,rt.guard);values=rt.request([job],p['initial'],'SHORT_ENGINEERING_NOT_NAVIGATION',full=True);a,row=values[0]
   if name=='baseline':base_arrays=a;R.F.M.audit_full_history(p,row,a)
   else:R.recovery_audit(p,row,a,base_arrays)
   rt.request([job],p['initial'],'EXACT_SHORT_ENGINEERING',full=True,reference=[row])
  finally:
   if rt.pool:rt.pool.close();rt.commands+=2
   R.fcntl.flock(rt.lock,R.fcntl.LOCK_UN);rt.lock.close()
  assert sum(rt.counts)==1000 and rt.commands==6;results.append(dict(kind=name,physics=sum(rt.counts),commands=rt.commands,rows=rt.rows,replays=rt.replays))
  save(folder/'summary.json',dict(state='COMPLETE',pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,new_Actor=0,new_Value=0))
 R.setup();save(OUT/'native_preflight.json',dict(state='PASS',pid=os.getpid(),physics=2000,commands=12,Actor=0,Value=0,actual_teacher_payload_locations_and_owner_audited=True,both_native_replays_full_arrays_exact=True,truncated_cap_NOT_navigation_success=True,results=results));print('T543_NATIVE_PROBE_PASS_2000_12',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
