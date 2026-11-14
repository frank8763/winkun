"""Registration and zero-native neural forward/load contracts, no real fitting."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t546_runtime as R
ROOT,OUT,DOC,A,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.A,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();previous=ROOT/'stage/runs/t545_protected_prefix_repair';s=read(previous/'summary.json');assert s['state']=='COMPLETE' and not s['result']['accepted'];assert read(previous/'audit.json')['state']=='PASS';R.G.stopped(previous);assert not Path('/proc',str(read(previous/'launch.json')['pid'])).exists();assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T545.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=read(previous/'protocol.json');cfg=read(p['actor_configuration']);source=R.G.rows(Path(p['audited_source_folder']))[:20];protected=[r for r in source if r['success'] and not r['collision']];assert len(protected)==19
 ids=p['initial_check']['field_success_ids'];assert len(ids)==12
 jobs=[cfg['old8_jobs'][6]]+[j for j in cfg['field_jobs'] if j['id'] in ids];assert len(jobs)==13 and jobs[0]['id']=='dev_task6_13'
 end=datetime(2026,11,14,12,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp()
 p.update(id='T546_FIXED_FUNCTIONAL_COMPATIBILITY',panel_jobs=jobs,protected_rows=protected,support_fraction=.49,minimum_feature_separation=1e-10,maximum_condition=1e8,rank_rtol=1e-10,max_bytes=8*2**20,child_bytes=8*2**20,model_bytes=2*2**20,compact_bytes=12*2**10,max_commands=30,preflight_physics=0,preflight_commands=0,main_seconds=2400,learning_seconds=300,audit_seconds=600,transport_seconds=600,source_seconds=180,absolute_all_stop=end,absolute_learning_stop=end-3600,absolute_native_stop=end-1800,scope='Finite feature compatibility engineering pilot. Frozen immutable full parentActor, one closed-form compact RBF residual plus exact shadow; independent rebuild audit. No new Adam/PPO/Value. Full original oldLEFT then12knownfieldsuccess,each exact. NO complete old8/field24/275/six/eight/fresh/reload; NO promotion. Old full history19+prefix170 zero residual pointwise, teacher-owned46 float32 actions exact before native. Fixed geometry/no runtime teacher/map/router/waypoint. Native failure stop, no retries or grids.',authorization='User nextstep on2026-11-14, today12:00all; daily6.5Mphysics240cmdunchanged. New maximum30commands based onactualprevious210. CPU14/15aggregateonecore/GPUoff.')
 p['max_physics']=2*sum(j['case']['step_cap'] for j in jobs)
 totals=[read(ROOT/'stage/runs'/f/'summary.json') for f in ['t543_feedback_joint_repair','t544_terminal_feedback_repair','t545_protected_prefix_repair']];prior_physics=sum(v['physics'] for v in totals)+3000;prior_commands=sum(v['commands'] for v in totals)+18
 assert prior_physics+p['max_physics']<=6500000 and prior_commands+p['max_commands']<=240
 assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<end and time.time()+p['main_seconds']<p['absolute_native_stop']
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [p['actor_configuration'],DOC/'T546_PLAN.md',DOC/'T546_FUNCTIONAL_COMPATIBILITY_SOURCES.md',DOC/'T545_STABILITY_PLASTICITY_REVIEW.md',previous/'protocol.json',previous/'summary.json',previous/'audit.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T545.json',p['recovery_row']['path'],Path(p['audited_source_folder'])/'cases.jsonl',Path(p['audited_source_folder'])/'audit.json']:p['inputs'][str(f)]=sha(f)
 for r in source:p['inputs'][r['path']]=r['sha256']
 global_before=R.used(ROOT/'stage/runs');assert global_before+p['max_bytes']+131072<2**30;OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 save(DOC/'T546_BUDGET.json',dict(main_physics=p['max_physics'],main_commands=30,prior_actual_physics=prior_physics,prior_actual_commands=prior_commands,day_worst_physics=prior_physics+p['max_physics'],day_worst_commands=prior_commands+30,day_limits=[6500000,240],own_bytes=p['max_bytes'],global_worst=global_before+p['max_bytes']+131072,global_limit=2**30,storage_contract=dict(full=1,full_upper_bytes=2**20,model_upper_bytes=2*2**20,compact_files=12,compact_each_bytes=12288,all_metadata_and_logs_bytes=4*2**20),native_scope='oldLEFT+known12field,13first+13exact,2setup2shutdown; no new native preflight',new_Adam=0,new_RL=0,closed_form_live=1,closed_form_shadow=1,independent_rebuild=1,all_stop_BJT='2026-11-14 12:00',main_seconds=2400,audit_seconds=600,backup_seconds=600,source_seconds=180));print('T546_REGISTERED_NOT_LAUNCHED',flush=True)

def preflight():
 p=R.configuration(read(OUT/'protocol.json'));R.setup();R.verify(p,True);assert not (OUT/'preflight.json').exists();q0=A.LEGACY_FROZEN(p,p['initial']['path']);q=A.Actor(p,1);missing,extra=q.load_state_dict(q0.state_dict(),strict=False);assert set(missing)=={'repair_centers','repair_radii','repair_weights'} and not extra
 x=R.torch.tensor(R.E.N.S.arrays(p['recovery_row']['path'])['states'][:12]);loc=q.sequence(x);assert R.torch.equal(loc,q0.sequence(x));alter=x.clone();alter[6:]+=1.;assert R.torch.equal(loc[:6],q.sequence(alter)[:6]);assert not any(v.requires_grad for v in q.parameters())
 q.reset_noise();h=None;maximum=0.
 for i,state in enumerate(x):
  u,l,h,_=q.step(state[None],h);maximum=max(maximum,float((l-loc[i]).abs().max()))
 assert maximum<1e-10
 q.reset_noise();q.sample=True;q.rng=R.np.random.default_rng(546);u,l,h,_=q.step(x[:1]);assert R.torch.isfinite(R.torch.tensor(q.tape[-1]['logp']));assert R.np.isfinite(q.tape[-1]['raw_actions']).all();assert R.np.array_equal(q.tape[-1]['atom'].astype(bool),u[0].abs().numpy()==1)
 model=A.export(q,p,OUT/'discarded_preflight.pt.gz');assert A.frozen(p,model['path']).phash()==q.phash();Path(model['path']).unlink();assert sha(p['initial']['path'])==p['initial']['sha256']
 # Fixed compact basis zero support, negative probe can detect interference.
 z=R.torch.zeros((1,184),dtype=R.torch.float64);c=z+2.;rad=R.torch.ones(1,dtype=R.torch.float64);assert float(A.basis(z,c,rad).max())==0. and float(A.basis(c,c,rad).min())==1.
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),new_native=0,new_commands=0,new_Adam=0,real_linear_fit=0,zero_residual_parent_sequence_exact=True,causal_prefix=True,step_sequence_max_error=maximum,density_boundary_atoms_finite=True,serialization_new_architecture_exact=True,immutable_parent_unmodified=True,new_worker_model_loading_NOT_yet_physically_exercised=True,same_verified_native_dynamics_worker_patch_loader_only=True));print('T546_PREFLIGHT_PASS_ZERO_NATIVE',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:save(DOC/('T546_'+sys.argv[1]+'_ERROR.json'),dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));raise
