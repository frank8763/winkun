"""Bounded old-recovery/new-success rehearsal registration and zero-native checks."""
from pathlib import Path
import datetime,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));import t531_runtime as R
G,L,P,torch,np,read,save,sha,digest=R.G,R.L,R.P,R.torch,R.np,R.read,R.save,R.sha,R.digest
OUT=R.OUT;DOC=ROOT/'stage/docs/day_20261113'
def prepare():
 assert not OUT.exists();prior=ROOT/'stage/runs/t530_early_bidirectional_recovery';assert read(prior/'pilot/audit.json')['state']=='PASS'
 assert all(not Path('/proc',str(i)).exists() for i in read(prior/'SEALED.json')['scientific_pids']+[read(prior/'launch.json')['pid']])
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T530.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=read(prior/'protocol.json');parent=p['neural_teacher'];initial=p['initial'];controls=[p['old_control_row'],p['new_control_row']];new=p['new_success_job']
 end=datetime.datetime(2026,11,13,17,30,tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp();assert time.time()+3120<end
 jobs=p['old8_jobs']+p['field_jobs'];assert len(jobs)==32;physics=2*sum(j['case']['step_cap'] for j in jobs)
 p.update(id='T531_JOINT_VERIFIED_SUCCESS_POLICY_DISTILLATION',out=str(OUT),initial=initial,initial_joint_locked=True,training_initial_joint_hash=initial['joint_hash'],epochs=32,evaluation_jobs=jobs,max_bytes=4*2**20,preflight_commands=6,pilot_commands=68,pilot_max_physics=physics,root_max_physics=physics,root_max_commands=74,root_max_main_seconds=2400,absolute_learning_stop=end-1800,absolute_native_stop=end-900,absolute_all_stop=end,
 scientific_scope='Joint distillation of actual successful source trajectories old8+field11 T507 and one newgoal T527. Complete causal histories and actual source locations; three equal family weights, halfmean/halftop20 per history. 32wholeActor Adam+shadow; source T527 wholeActor/twoAdamgroups retained,Value0/PG0. No new teacher recovery queries or source-control simulations. Standard policy consolidation engineering, not original algorithm, exact CLEAR/DAgger reproduction, or new onpolicy RL.',
 acceptance='Same autonomous singleActor old8all8 and fullfield24 retaining alloriginal11fieldIDs+newgoal,field>=12/contact<=9/noinvalidcensor. Else wholeActorAdamrollback initial research state; original acceptedT507 staysunchanged. No model selection from fresh data or inference teacher/task router.',
 authorization='User explicitly continue experiment within confirmed17:30BJT bounded window. Failed T530 ended audited/backed, not restarted or extended. New fixed objective removes unsupported off-state recovery labels and uses only true successful source functions; no grid or extra seeds.')
 for k in ['recovery_row','baseline_row','initial_protected_ids','initial_field_success_ids','zero_native_registration_accounting_fix','zero_native_audit_reporting_fix','divergence','teacher_first','prefix_physics_steps']:p.pop(k,None)
 p['inputs']={f:h for f,h in p['inputs'].items() if Path(f).suffix=='.py'}
 p['protected_files']={initial['path']:initial['sha256'],parent['path']:parent['sha256'],str(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz'):sha(ROOT/'stage/runs/t255_overnight/s2_reset_final_checkpoint.pt.gz')}
 for f in [*HERE.glob('*.py'),DOC/'T531_PLAN.md',prior/'pilot/summary.json',prior/'pilot/audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T530.json']:
  p['inputs'][str(f)]=sha(f)
 for r in controls:p['inputs'][r['path']]=r['sha256']
 for r in p['keep_rows']:p['inputs'][r['path']]=r['sha256']
 p['inputs'].update(p['protected_files']);obj=L.make(p);assert digest(P.state(*obj))==initial['joint_hash']
 reserve=p['model_bytes']+32*p['compact_bytes']+int(1.25*2**20);used=G.M.used();assert reserve<p['max_bytes'] and used+p['max_bytes']+65536<2**30,(used,reserve)
 OUT.mkdir();blob=json.dumps(p,ensure_ascii=False,separators=(',',':'))+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
 save(DOC/'T531_BUDGET.json',dict(max_physics=physics,max_commands=74,main_seconds=2400,learning_seconds=1200,audit_seconds=300,backup_reserve=300,all_stop=end,own_bytes=p['max_bytes'],data_model_metadata_reserve=reserve,global_reserved=used+p['max_bytes'],live_Adam=32,shadow_Adam=32,audit_historical_Adam=32,preflight_discarded_Adam=4,newPG=0,ValueAdam=0,CPU=[14,15],CPUQuota='100%',GPU=False));print('T531_REGISTERED_NOT_LAUNCHED',flush=True)
def preflight():
 p=read(OUT/'protocol.json');R.verify(p,True);assert not (OUT/'preflight.json').exists();P.frozen=L.A.frozen;obj=L.make(p);shadow=L.make(p);before=P.state(*obj);refs=L.references(p)
 fixture=[dict(g,x=g['x'][:8],target=g['target'][:8]) for g in refs]
 aa=L.fit(obj,p,fixture,lambda:None,lambda *_:None,steps=2);bb=L.fit(shadow,p,fixture,lambda:None,lambda *_:None,steps=2);assert aa==bb and digest(P.state(*obj))==digest(P.state(*shadow));model=L.A.export(obj,OUT/'discarded_preflight_clone.pt.gz',p['model_bytes']);assert digest(P.checkpoint(model['path'])['joint_state'])==model['joint_hash']
 for z in [obj,shadow]:P.load(z,before);assert digest(P.state(*z))==p['initial']['joint_hash']
 clone=dict(model);Path(model['path']).unlink();folder=OUT/'preflight_native';folder.mkdir();counts=G.R.M.mp.get_context('spawn').RawArray('q',3);pool=R.F.Pool(dict(p,phase='student',out=str(folder),native_stop_epoch=time.time()+120,max_physics=1),counts,lambda:None)
 try:
  vals=pool.request([dict(model=p['initial']['path'],job=p['old8_jobs'][6],seed=None,cancel_probe=True)]);assert vals[0]['state']=='CANCEL_PASS' and sum(counts)==0
 finally:pool.close()
 # Two initializations + one cancel request + two close commands = five;
 # retain conservative registered six command budget, record actual five.
 save(OUT/'preflight.json',dict(state='PASS',pid=os.getpid(),native=0,commands_actual=5,commands_budget=6,discarded_synthetic_Actor_Adam=4,actual_singleActor_worker_cancel=True,actual19_parent_and_one_newgoal_success_targets_verified=True,truncated_actual_success_history_fixture_NOT_navigation=True,full_ActorAdam_live_shadow_serialization_rollback_exact=True,discarded_clone=clone));print('T531_PREFLIGHT_PASS',flush=True)
def launch():
 p=read(OUT/'protocol.json');R.verify(p,True);pre=read(OUT/'preflight.json');assert pre['state']=='PASS' and not Path('/proc',str(pre['pid'])).exists() and not (OUT/'launch.json').exists();src=read(ROOT/'stage/runs/t224_archive/latest_source_sync_receipt.json');assert src['root_tree']==subprocess.check_output(['git','rev-parse','HEAD^{tree}'],cwd=ROOT,text=True).strip() and not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip();seconds=int(p['absolute_all_stop']-time.time());assert seconds>3120
 subprocess.run(['systemd-run','--user','--unit=stage-t531-joint-success-distillation','--collect','-p','CPUAffinity=14 15','-p','CPUQuota=100%','-p','CPUWeight=1','-p','IOWeight=1','-p','Nice=19','-p','CPUSchedulingPolicy=idle','-p','RuntimeMaxSec='+str(seconds),'-p','TimeoutStopSec=3','-p','KillMode=control-group','-p','Environment=CUDA_VISIBLE_DEVICES=','-p','Environment=OPENBLAS_NUM_THREADS=1','-p','Environment=OMP_NUM_THREADS=1','-p','Environment=MKL_NUM_THREADS=1','-p','WorkingDirectory='+str(ROOT),'-p','StandardOutput=append:'+str(OUT/'driver.log'),'-p','StandardError=append:'+str(OUT/'driver.log'),'systemd-inhibit','--what=sleep:idle','--mode=block','--who=Stage-T531','--why=Joint verified success policy distillation','taskset','-c','14',str(ROOT/'run_stage.sh'),str(HERE/'driver.py')],check=True);print('T531_SERVICE_SUBMITTED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
