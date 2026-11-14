"""Lock eight tasks, three immutable models and four physical perturbations."""
from pathlib import Path
import copy,sys,time,os
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t551_runtime as R
ROOT,OUT,DOC,L,np,torch,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.np,R.torch,R.read,R.save,R.sha,R.digest

def prepare():
 assert not OUT.exists();old=ROOT/'stage/runs/t550_offline_obstacle_relay';source=read(old/'protocol.json');report=read(DOC/'T550_FINAL_REPORT.json');assert report['state']=='COMPLETE' and report['remote']=='PENDING_OFFLINE';assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T550.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED';assert not Path('/proc',str(read(old/'launch.json')['pid'])).exists();assert read(DOC/'T550_BACKUP_RECOVERY.json')['remote']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 ids=['dev_task7_07','dev_task7_25','dev_task7_26','hold_task6_04','hold_task7_06','hold_task7_07','t502_field_sector1_offset2','t502_field_sector4_offset2'];by={j['id']:j for j in source['final_jobs']};jobs=[by[i] for i in ids];t549=ROOT/'stage/runs/t549_curvature_gated_transfer';oldfinal=ROOT/'stage/runs/t542_offline_field_relay/final_selected407';models=dict(T507=read(oldfinal/'protocol.json')['initial'],T549=read(old/'protocol.json')['initial'],T550=report['retained_model']);historical={}
 for name,f in [('T507',oldfinal/'cases.jsonl'),('T549',t549/'cases.jsonl'),('T550',old/'final407/cases.jsonl')]:
  historical[name]={r['job_id']:r for r in [__import__('json').loads(s) for s in f.read_text().splitlines()] if r['job_id'] in ids and r['policy_hash']==models[name]['policy_hash']}
 pulse=[];witness=[]
 for child,ident in [('attempt_00_gap',ids[-2]),('attempt_01_gap',ids[-1])]:
  row=R.G.rows(old/child)[1];assert row['job_id']==ident and row['success'] and not row['collision'];assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);n=len(a['states']);hits=[k for k in range(max(1,n//3),2*n//3) if abs(float(a['actions'][k-1,1]))<=.9];assert hits;k=hits[0];witness.append(dict(id=ident,row=row,hold=k,seconds=float(a['native_counts'][:k].sum()*.002)))
  for sign in [-1,1]:
   job=copy.deepcopy(by[ident]);job.update(id=ident+('__steer_minus' if sign<0 else '__steer_plus'),baseline_id=ident,pulse_hold=k,pulse_sign=sign);pulse.append(job)
 auth=read(DOC/'T551_WINDOW_AUTHORIZATION.json');physics=6*sum(j['case']['step_cap'] for j in jobs)+2*sum(j['case']['step_cap'] for j in pulse);assert physics==auth['max_physics']==2122264
 p=dict(id='T551_FROZEN_RETENTION_ATTRIBUTION_AND_REAL_PULSE_RECOVERY',actor_configuration=source['actor_configuration'],source_protocol=source['source_protocol'],initial=models['T550'],models=models,jobs=jobs,pulse_jobs=pulse,pulse_witness=witness,lost6_ids=ids[:6],historical_rows=historical,raw_sigma=source['raw_sigma'],field_limit_m=source['field_limit_m'],max_bytes=32*2**20,child_bytes=32*2**20,full_bytes=2**20,compact_bytes=12288,max_physics=physics,max_commands=72,main_seconds=1800,learning_seconds=0,audit_seconds=300,transport_seconds=300,source_seconds=120,absolute_all_stop=auth['all_stop_epoch'],absolute_native_stop=auth['native_stop_epoch'],absolute_learning_stop=auth['observed_reply_epoch'],new_Adam=0,new_RL=0,scope='Fixed8tasks3models24full+24exact and4controlledsteerpulses+4exact. One .2s executionsteer+.1/- .1normalized (requested2.5deg), driveunchanged, genuineprefix/clock/history/hidden,noreset. Pulsepointfirstmiddlethirdprevioussteerunsaturated. Notexpertdata,notonpolicy,notcrosssite, nolearningorparametergrid. Fullhistoryfillsmissingcausalcomparisons; stoponcontractfail,norestart.',protected_files={m['path']:m['sha256'] for m in models.values()})
 # Delta checkpoints load immutable parent layers, all kept locally and locked.
 m=models['T550']
 while 'parent' in m:m=m['parent'];p['protected_files'][m['path']]=m['sha256']
 p['inputs']=dict(p['protected_files'])
 for module in list(sys.modules.values()):
  f=getattr(module,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in HERE.glob('*.py'):p['inputs'][str(f)]=sha(f)
 for f in [source['actor_configuration'],DOC/'T551_PLAN.md',DOC/'T551_WINDOW_AUTHORIZATION.json',DOC/'T550_TASKWISE_REGRESSION_REVIEW.json',DOC/'ROBUSTNESS_PRIMARY_REVIEW_20261114.md',DOC/'CROSS_SITE_RESEARCH_PLAN.md',old/'protocol.json',t549/'protocol.json',DOC/'T550_FINAL_REPORT.json',ROOT/'stage/docs/long_distance_20261101/release_receipt_T550.json',oldfinal/'cases.jsonl',t549/'cases.jsonl',old/'final407/cases.jsonl']:p['inputs'][str(f)]=sha(f)
 for w in witness:p['inputs'][w['row']['path']]=w['row']['sha256']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<=2**30;assert time.time()+p['main_seconds']+p['audit_seconds']+p['transport_seconds']+p['source_seconds']+90<p['absolute_all_stop'];assert time.time()+p['main_seconds']<p['absolute_native_stop'];OUT.mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p);R.verify(p,True)
 save(DOC/'T551_BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',max_physics=physics,max_commands=72,day_prior_physics=20990558,day_prior_commands=1032,day_worst_physics=20990558+physics,day_worst_commands=1104,day_max_physics=auth['day_max_physics'],day_max_commands=auth['day_max_commands'],own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,full_payloads28=28*2**20,metadata_partial=4*2**20,model_outputs=0,new_native_engineering=0,new_optimizer=0,all_stop_epoch=p['absolute_all_stop'],native_stop_epoch=p['absolute_native_stop']));print('T551_REGISTERED_NOT_LAUNCHED')

def preflight():
 p=R.configuration(read(OUT/'protocol.json'));R.setup();R.verify(p,True);assert not (OUT/'preflight.json').exists();a=R.E.N.S.arrays(p['pulse_witness'][0]['row']['path']);x=torch.tensor(a['states'][:12])
 for name,m in p['models'].items():
  q=L.frozen(p,m['path']);assert q.phash()==m['policy_hash'];q.sample=False;q.reset_noise();base=q.sequence(x);future=x.clone();future[6:]+=1;assert torch.equal(base[:6],q.sequence(future)[:6]);h=None
  for state in x:
   u,l,h,_=q.step(state[None],h);assert np.isfinite(u.numpy()).all() and u.shape==(1,2)
  assert sha(m['path'])==m['sha256']
 for sign in [-1,1]:
  u=torch.tensor([[.5,.2]],dtype=torch.float32);v=R.perturbed_command(u,sign);assert torch.equal(u[:,0],v[:,0]) and abs(float(v[0,1]-u[0,1])-sign*.1)<1e-6 and torch.equal(u,torch.tensor([[.5,.2]],dtype=torch.float32))
 for sign,u in [(0,torch.zeros(1,2)),(1,torch.tensor([[0.,.99]]))]:
  try:R.perturbed_command(u,sign)
  except AssertionError:pass
  else:raise AssertionError('Invalid pulse accepted')
 # Raw array allocation before compression is measured from real payload schema;
 # per-file and aggregate guards remain in native worker and owner runtime.
 assert len(p['jobs'])==8 and len(p['pulse_jobs'])==4 and len(p['models'])==3
 save(OUT/'preflight.json',dict(state='PASS_ZERO_NATIVE',pid=os.getpid(),new_native=0,new_optimizer=0,all3_real_models_hash_reload_and_causal_step=True,pulse_changes_only_steer=True,pulse_drive_and_source_tensor_unchanged=True,invalid_sign_and_saturation_rejected=True,source_native_full_history_contract_reused=True,actual_new_pulse_session_NOT_native_validated=True,not_navigation_or_robustness_success=True));print('T551_ZERO_NATIVE_PASS')
if __name__=='__main__':globals()[sys.argv[1]]()
