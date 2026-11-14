"""Register a single-factor addition after frozen saved-evidence diagnosis."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import copy,os,sys,time,traceback
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t545_runtime as R
ROOT,OUT,DOC,L,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();old=ROOT/'stage/runs/t544_terminal_feedback_repair';s=read(old/'summary.json');assert s['state']=='COMPLETE' and s['result']['teacher_qualified'] and not s['result']['accepted'];assert read(old/'audit.json')['state']=='PASS';R.G.stopped(old);assert not Path('/proc',str(read(old/'launch.json')['pid'])).exists()
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T544.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';diagnosis=read(DOC/'T544_POSTHOC_INTERFERENCE.json');assert diagnosis['state']=='SAVED_INTERFERENCE_CONFIRMED' and diagnosis['prefix']['first_policy_action_change_hold']==0
 p=read(old/'protocol.json');end=datetime(2026,11,14,12,0,tzinfo=ZoneInfo('Asia/Shanghai')).timestamp();assert time.time()+5400<end;teacher=R.G.rows(old)[0];assert teacher['success'] and not teacher['collision'];a=R.E.N.S.arrays(teacher['path']);assert R.np.flatnonzero(a['ownership'])[0]==170
 p.update(id='T545_ONE_FACTOR_PROTECTED_STUDENT_PREFIX',recovery_row=teacher,fixed_intervention=read(old/'derived_prefix.json'),teacher_first=170,prefix_keep_weight=1.,max_bytes=8*2**20,child_bytes=8*2**20,max_commands=68,preflight_physics=0,preflight_commands=0,absolute_all_stop=end,absolute_learning_stop=end-3600,absolute_native_stop=end-1800,scope='Only add prefix latent preservation to recovery objective and consequently virtual MIR proposal; lambda1 GaussianKLlike existing memories,old.25field.25/recovery.5 unchanged. SameT542rtg06 initial freshAdam, sameT544 verifiedsuccessfulteacher history, same19success histories and16live+16shadow eachwith16virtualdiscarded. No teacher recollection ornewdata. Full singleActorold8+field24+32exact, original12/old8/contact<=9 gate unchanged;wholeActorAdam rollback, no grids/restart/newRL/Value. Initialprefixpenaltygradientzero isnotfirststepguarantee.',authorization='2026-11-14user asks thoroughlydiagnose thencontinue,12:00allpreviouslyconfirmed. Todaycumulativeworst<=6.5Mphysics/240commands,own8MiB/global1GiB/free40GiB CPU14/15aggregateonecore GPUoff.')
 base=read(p['actor_configuration']);p['max_physics']=2*sum(j['case']['step_cap'] for j in base['old8_jobs']+base['field_jobs']);t543=read(ROOT/'stage/runs/t543_feedback_joint_repair/summary.json');daily=t543['physics']+2000+s['physics']+1000+p['max_physics'];assert daily<=6500000;commands=t543['commands']+12+s['commands']+6+p['max_commands'];assert commands<=240
 p['inputs']=dict(p['protected_files'])
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [p['actor_configuration'],DOC/'T545_PLAN.md',DOC/'T545_PREFIX_PRESERVATION_SOURCES.md',DOC/'T544_POSTHOC_INTERFERENCE.json',old/'protocol.json',old/'summary.json',old/'audit.json',old/'native_preflight.json',old/'cases.jsonl',ROOT/'stage/docs/long_distance_20261101/release_receipt_T544.json',teacher['path'],Path(p['audited_source_folder'])/'cases.jsonl',Path(p['audited_source_folder'])/'audit.json']:p['inputs'][str(f)]=sha(f)
 for r in R.G.rows(Path(p['audited_source_folder']))[:20]:p['inputs'][r['path']]=r['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30;OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 save(DOC/'T545_BUDGET.json',dict(main_physics=p['max_physics'],main_commands=68,new_teacher_native=0,new_engineering_native=0,new_engineering_commands=0,day_cumulative_physical_worst=daily,day_physical_limit=6500000,day_cumulative_commands_worst=commands,day_command_limit=240,liveAdam=16,live_virtual_discardedAdam=16,shadowAdam=16,shadow_virtualAdam=16,audit_historical_discardedAdam=16,audit_virtualAdam=16,synthetic_preflight_discardedAdam=8,new_PG=0,Value=0,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes'],global_limit=2**30,main_seconds=p['main_seconds'],audit_seconds=p['audit_seconds'],backup_seconds=p['transport_seconds'],source_seconds=p['source_seconds'],all_stop_BJT='2026-11-14 12:00'));print('T545_REGISTERED_NOT_LAUNCHED',daily,flush=True)

def preflight():
 p=R.configuration(read(OUT/'protocol.json'));R.setup();R.verify(p,True);assert not (OUT/'preflight.json').exists();obj=L.make(p);shadow=L.make(p);before=L.state(obj);q=obj[0];a0=R.E.N.S.arrays(p['recovery_row']['path']);a={k:v for k,v in a0.items()};a.update(states=a0['states'][166:178],teacher_actions=a0['teacher_actions'][166:178],locations=a0['locations'][166:178],ownership=a0['ownership'][166:178],stochastic=a0['stochastic'][166:178])
 # Short fixture's history starts at h0; its own locations are synthetic, not
 # claimed to be a real reset or new navigation data.
 x=R.torch.tensor(a['states'])
 with R.torch.no_grad():target=q.sequence(x).detach().clone()
 a['locations']=target.numpy().copy();refs=[dict(id=('old' if i<7 else 't502_field')+str(i).zfill(2),family=int(i>=7),x=x,target=target.clone()) for i in range(19)]
 import importlib.util
 spec=importlib.util.spec_from_file_location('_t545_previous_loss',ROOT/'stage/experiments/t544_terminal_feedback_repair/t544_learning.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 zero=dict(p,prefix_keep_weight=0.);newloss=L.repair(q,a,zero);oldloss=old.repair(q,a,p);assert R.torch.equal(newloss,oldloss);g0=R.torch.autograd.grad(newloss,list(q.parameters()));g1=R.torch.autograd.grad(oldloss,list(q.parameters()));assert all(R.torch.allclose(u,v,atol=1e-12,rtol=0) for u,v in zip(g0,g1))
 with R.torch.no_grad():tail,pref=L.loss_parts(q,a,p);assert float(pref)<1e-20;alter=x.clone();alter[4:]+=1;assert R.torch.equal(q.sequence(x)[:4],q.sequence(alter)[:4])
 cfg=dict(p,fit_steps=2);aa=L.fit(obj,a,cfg,refs,lambda:None,lambda *_:None);bb=L.fit(shadow,a,cfg,refs,lambda:None,lambda *_:None);assert aa==bb and digest(L.state(obj))==digest(L.state(shadow));model=L.export(obj,p,OUT/'discarded_preflight.pt.gz');assert L.A.frozen(p,model['path']).phash()==model['policy_hash'];Path(model['path']).unlink()
 assert aa['trace'][0]['protected_prefix_KL']<1e-20 and aa['trace'][1]['protected_prefix_KL']>0
 for o in [obj,shadow]:L.load(o,before);assert digest(L.state(o))==digest(before)
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),new_native=0,discarded_synthetic_Adam=4,discarded_synthetic_virtual_Adam=4,lambda0_old_loss_and_gradient_equivalence=True,causal_GRU_prefix_unaffected_by_future_inputs=True,initialprefix_penalty_zero_then_positive=True,fullActorAdam_virtual_and_shadow_rollback_exact=True,model_reload_exact=True));save(OUT/'native_preflight.json',dict(state='REUSED_VERIFIED_SAME_NATIVE_CONTRACT',new_native=0,new_commands=0,actual_source_native_preflight=str(ROOT/'stage/runs/t544_terminal_feedback_repair/native_preflight.json'),source_audit_sha256=sha(ROOT/'stage/runs/t544_terminal_feedback_repair/audit.json'),same_Actor_architecture_and_normal_cpu_worker_and_native_dynamics=True,teacher_raw_reused_not_recollected=True));print('T545_PREFLIGHT_PASS_ZERO_NATIVE',flush=True)
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException:save(DOC/('T545_'+sys.argv[1]+'_ERROR.json'),dict(error=traceback.format_exc(),raw_preserved=True,no_restart=True));raise
