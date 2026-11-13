"""Bounded old-recovery/new-success rehearsal registration and zero-native checks."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t529_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
 assert not OUT.exists();prior=ROOT/'stage/runs/t528_module_transfer_probe';assert read(prior/'pilot/audit.json')['state']=='PASS'
 assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T528.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 source=ROOT/'stage/runs/t527_validated_recovery_learning';p=read(source/'protocol.json');initial=read(source/'pilot/summary.json')['result']['model'];parent=p['initial'];old=next(j for j in p['old8_jobs'] if j['id']=='dev_task6_13');new=next(j for j in p['field_jobs'] if j['id']=='t502_field_sector0_offset2')
 rows=G.rows(source/'pilot');controls=[next(r for r in rows if r['job_id']==j['id']) for j in [old,new]];assert controls[0]['collision'] and controls[1]['success'] and all(r['policy_hash']==initial['policy_hash'] for r in controls)
 end=datetime.datetime(2026,11,13,17,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp();assert time.time()+2520<end
 jobs=p['old8_jobs']+p['field_jobs'];assert len(jobs)==32;physics=2*(old['case']['step_cap']+new['case']['step_cap'])+2*old['case']['step_cap']+2*sum(j['case']['step_cap'] for j in jobs)
 p.update(id='T529_BIDIRECTIONAL_REAL_RECOVERY_AND_NEW_SUCCESS_REHEARSAL',out=str(OUT),initial=initial,initial_joint_locked=True,training_initial_joint_hash=initial['joint_hash'],neural_teacher=parent,locked_best=parent,parent_model=parent,probe_job=old,new_success_job=new,source_controls=controls,teacher_first=151,prefix_physics_steps=15100,epochs=16,evaluation_jobs=jobs,baseline_field_success_ids=[r['job_id'] for r in p['keep_rows'] if r['job_id'].startswith('t502_')],full_bytes=2**20,compact_bytes=8192,model_bytes=2*2**20,max_bytes=7*2**20,preflight_commands=6,pilot_commands=82,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=88,root_max_main_seconds=1800,absolute_learning_stop=end-1800,absolute_native_stop=end-900,absolute_all_stop=end,
 scientific_scope='Rebuild T527 actual old-left failure and autonomous newgoal success with full120/history/native exact controls. From actual old failure prefix30.2s, one fixed T507 neural recovery with independently carried teacher hidden, original clock/goal/history. Fullteacher safeGOAL qualifies labels; no untrained student handoff gate. Then16wholeActor supervised updates+shadow with old8/field11/actualT527newgoal successful location rehearsal equally weighted by family. No newPG/Value; joint engineering pilot, not novel CLEAR/DAgger or causal module ablation.',
 acceptance='One updated singleActor autonomous old8all8 plus fullfield24; retain all original11fieldIDs AND newtarget,field>=12/contact<=9/noinvalidcensor. Else entireActorAdamrollbackT527researchinitial, original acceptedT507 unchanged. Noold275/six54/eight54/fresh16/78.',
 optimizer_contract='Full T527 wholeActor and BOTH Adam groups/moments/steps inherited exactly, lr3e-5 unchanged. Value weights+Adam immutable. No architecture/capacity/reward/cap/radius changes.',authorization='User explicitly confirmed new bounded window allstop2026-11-13 17:30BJT. Training-only teachers authorized; deployment singleActor no planner/map/waypoint/runtime teacher/router. No old run restart or noise/seed grids.')
 for k in ['recovery_row','baseline_row','initial_protected_ids','initial_field_success_ids']:p.pop(k,None)
 p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
 p['protected_files']={initial['path']:initial['sha256'],parent['path']:parent['sha256'],str(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'):sha(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz')}
 for f in [*HERE.glob('*.py'),DOC/'T529_PLAN.md',DOC/'T529_STORAGE_LEDGER.json',source/'pilot/summary.json',source/'pilot/audit.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T527.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T528.json']:
  p['inputs'][str(f)]=sha(f)
 for r in p['keep_rows']:p['inputs'][r['path']]=r['sha256']
 p['inputs'].update(p['protected_files']);obj=L.make(p);assert digest(P.state(*obj))==initial['joint_hash']
 reserve=3*p['full_bytes']+p['model_bytes']+32*p['compact_bytes']+int(1.25*2**20);used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30,(used,reserve)
 OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
 save(DOC/'T529_BUDGET.json',dict(max_physics=physics,max_commands=88,main_seconds=1800,learning_seconds=1200,audit_seconds=300,backup_reserve=300,all_stop=end,own_bytes=p['max_bytes'],data_model_metadata_reserve=reserve,global_reserved=used+p['max_bytes'],live_Adam=16,shadow_Adam=16,audit_historical_Adam=16,preflight_discarded_Adam=4,newPG=0,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False));print('T529_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
 p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();P.frozen=L.A.frozen;obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=[]
 with torch.no_grad():
  for row in p['keep_rows']:
   a=G.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);target=torch.tensor(a['locations']);source=L.A.frozen(p,row['request']['model']);assert float((source.sequence(x)-target).abs().max())<1e-10;refs.append(dict(x=x,target=target,family=int(row['job_id'].startswith('t502_'))))
  x=refs[6]['x'][:8];refs.append(dict(x=x,target=obj[0].sequence(x).detach().clone(),family=2,synthetic_fixture_NOT_new_success=True));teacher=L.A.frozen(p,p['neural_teacher']['path']);th=None;labels=[]
  for state in x:
   u,_,th,_=teacher.step(state[None],th);labels.append(u[0].numpy())
 a=dict(states=x.numpy(),teacher_actions=np.asarray(labels),ownership=np.asarray([0]*4+[1]*4,np.uint8),stochastic=np.zeros(8,np.uint8))
 aa=L.fit(obj,a,p,refs,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,a,p,refs,lambda:None,lambda *_:None,steps=2);assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
 for z in [obj,shadow]:P.load(z,before);assert digest(P.state(*z))==p['initial']['joint_hash']
 clone=dict(model);Path(model['path']).unlink();folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.Recovery.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
 try:
  vals=pool.request([dict(model=p['initial']['path'],job=dict(p['probe_job'],teacher_holds=-1),seed=None,cancel_probe=True)]);assert vals[0]['state']=='CANCEL_PASS' and sum(counts)==0
 finally:pool.close()
 # Two initializations + one cancel request + two close commands = five;
 # retain conservative registered six command budget, record actual five.
 save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands_actual=5,commands_budget=6,discarded_synthetic_Actor_Adam=4,independent_teacher_carry_and_actual_model_cancel=True,actual19_parent_targets_verified=True,newgoal_reference_synthetic_only_NOT_navigation=True,full_ActorAdam_live_shadow_serialization_rollback_exact=True,discarded_clone=clone));print('T529_PREFLIGHT_PASS',flush=True)
def launch():
 p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();seconds=int(p['absolute_all_stop']-time.time());assert seconds>2520
 subprocess.run(['systemd-run','--user','--unit=stage-t529-bidirectional-recovery','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T529','--why=Bidirectional validated old recovery and actual new success rehearsal','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True);print('T529_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
