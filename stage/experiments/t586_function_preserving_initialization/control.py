"""Preparation only until a fresh user-approved window; preflight has zero Adam."""
from pathlib import Path
import ast,copy,datetime,importlib.util,json,sys,time
import t586_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
 old=ROOT/'stage/runs/t584_unbounded_recovery_learning';assert read(old/'audit.json')['state']=='PASS';assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids']);assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T584.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 p=copy.deepcopy(read(old/'protocol.json'));v=read(old/'summary.json')['result'];keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];cap=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+keys)+2*sum(j['case']['step_cap'] for j in p['handoff_jobs'])
 epoch=lambda t:datetime.datetime.fromisoformat('2026-11-16T'+t+':00+08:00').timestamp()
 p.update(id='T586_ANCHORED_NONZERO_INITIALIZATION_RECOVERY_PILOT',out=str(OUT),historical_A_model=v['proposal'],historical_A_fit=v['update']['after'],historical_handoff_baseline=v['baseline_handoff'],
  BC_lr=.0003,BC_steps=100,keep_weight=100.,main_seconds=780,learning_seconds=240,audit_seconds=240,transport_seconds=180,source_seconds=90,
  absolute_learning_stop=epoch('13:40'),absolute_native_stop=epoch('13:50'),absolute_all_stop=epoch('14:00'),
  max_physics=cap,max_commands=63,max_bytes=18*2**20,child_bytes=18*2**20,model_bytes=512*1024,inputs={},protected_files={},
  new_Actor_Adam_cap=100,shadow_Adam_cap=100,audit_discarded_Adam_cap=100,preflight_discarded_Adam_cap=0,new_RL=0,new_Critic=0,
  authorization_approved=False,user_authorization='2026-11-16 user: 下一轮开始; proposed new14:00 window PENDING, previous13:00 ended',
  scope='Historical A reused, only new B;100BC+100shadow and independent100Adam audit;offline strict3teacherMAE vsA and27oldhistorymax<=A gate, else0native. Conditional3learnedhandoff+49original autonomous+7exact; no grid/retry/next mechanism.',
  count_contract='T586 explicit fields only; inherited loading compatibility fields not operational')
 upper=9*p['full_bytes']+45*p['compact_bytes']+3*p['model_bytes']+3*2**20;assert upper<=p['max_bytes'];glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
 R.setup();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.dataset(p);q=L.fresh(p);a=L.Prior.fresh(p);initial_hash=digest(q.state_dict());checks=0
 # All27 keep and3 teacher histories, using the frozen real GRU and current120 inputs.
 for row in p['keep_rows']+p['recovery_rows']:
  z=R.E.N.S.arrays(row['path']);x=torch.tensor(z['states'])
  with torch.no_grad():
   f,b=L.feature(q,x);assert torch.equal(L.loc(q,f,b),b) and torch.equal(q.sequence(x),a.sequence(x))
  checks+=1
 for row in v['baseline_handoff']:
  z=R.E.N.S.arrays(row['path']);h=ha=None
  with torch.no_grad():
   for i,x in enumerate(z['states'][:16]):
    u,loc,h,_=q.step(torch.tensor(x[None]),h);ua,la,ha,_=a.step(torch.tensor(x[None]),ha);assert torch.equal(u,ua) and torch.equal(loc,la) and torch.equal(h,ha)
 # Actual first gradient, no optimizer object or mutation. Same teacher objective.
 q.continual.requires_grad_(True);a.continual.requires_grad_(True)
 def grads(model,location):
  loss=sum(.5*((location(model,g['f'],g['b'])-g['target'])/.2).square().mean()/len(data[1]) for g in data[1]);return torch.cat([x.flatten() for x in torch.autograd.grad(loss,list(model.continual.parameters()))])
 gb=grads(q,L.loc);ga=grads(a,L.Prior.loc);assert torch.count_nonzero(ga[:5920])==0 and torch.linalg.vector_norm(gb[:5920])>0
 assert digest(q.state_dict())==initial_hash and not any(x.requires_grad for x in q.anchor.parameters())
 temp=DOC/'discarded_no_optimizer.pt.gz';model=L.export((q,None),p,temp);assert L.frozen(p,temp).phash()==q.phash();temp.unlink()
 assert sum(x.numel() for x in q.continual.parameters())==5986 and sum(x.numel() for x in q.anchor.parameters())==5986
 spec=importlib.util.spec_from_file_location('_t586_transport',HERE/'transport.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert m.OUT==OUT
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_NOT_REGISTERED',native=0,optimizer_objects=0,optimizer_updates=0,actual_discarded_teacher_gradients=2,complete_histories_initial_function_exact=checks,step48_actions_locations_hidden_exact=True,
  first_layer_nonzero_gradient_A=int(torch.count_nonzero(ga[:5920])),first_layer_nonzero_gradient_B=int(torch.count_nonzero(gb[:5920])),trainable_parameters=5986,additional_frozen_reference_parameters=5986,not_total_parameter_matched=True,
  model_readback_exact=True,parameters_unchanged=True,transport_OUT_matches=True,own_storage_upper=upper,global_worst=glob,authorization='PENDING_NEW_WINDOW'))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 files=[*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),old/'audit.json',old/'summary.json',old/'SEALED.json',Path(v['proposal']['path'])]+[Path(r['path']) for r in p['keep_rows']+p['recovery_rows']+v['baseline_handoff']]
 for f in files:p['inputs'][str(f)]=sha(f)
 for j in p['handoff_jobs']:p['inputs'][j['source_path']]=sha(j['source_path'])
 m=p['base_model']
 while m:
  assert sha(m['path'])==m['sha256'];p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
 for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',new_window='PENDING14:00',max_physics=cap,max_commands=63,main_seconds=780,learning_seconds=240,audit_seconds=240,transport_seconds=180,source_seconds=90,all_work_reserve_seconds=1320,own_bytes=p['max_bytes'],own_storage_upper=upper,global_worst=glob,conditional_full7_compact45=True,live_Adam=100,shadow_Adam=100,audit_historical_Adam=100,preflight_Adam=0,no_science_run_directory_created=True))
 print('T586_PREPARED_PREFLIGHT_PASS_NOT_REGISTERED',cap,glob,flush=True)

def optimizer_preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json')
 assert auth['approved'] is True and auth['all_stop_epoch']==p['absolute_all_stop'] and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands'] and auth['own_bytes']==p['max_bytes']
 assert time.time()+1320<p['absolute_all_stop'];R.setup();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.dataset(p);a,b=L.make(p),L.make(p);before=L.state(a)
 u=L.fit(a,data,p,lambda:None,lambda *_:None,steps=2);v=L.fit(b,data,p,lambda:None,lambda *_:None,steps=2);assert u==v and digest(L.state(a))==digest(L.state(b));mature=L.state(a)
 L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
 temp=DOC/'discarded_authorized_mature.pt.gz';model=L.export(a,p,temp);assert L.frozen(p,temp).phash()==a[0].phash();temp.unlink()
 save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',new_native=0,discarded_Adam=4,live_shadow_mature_ActorAdam_rollback_exact=True,model_readback_exact=True,model_bytes=model['bytes']))
 p['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',p);print('T586_AUTHORIZED_OPTIMIZER_PREFLIGHT_PASS')

def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');assert read(DOC/'PREFLIGHT_OPTIMIZER.json')['state']=='PASS_AUTHORIZED_ZERO_NATIVE';auth=read(DOC/'AUTHORIZATION.json');assert auth['approved'] is True and auth['all_stop_epoch']==p['absolute_all_stop'] and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands'] and auth['own_bytes']==p['max_bytes'] and auth['user_reply']
 assert time.time()+1320<p['absolute_all_stop'] and time.time()+780<p['absolute_native_stop'] and time.time()+240<p['absolute_learning_stop'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 p.update(authorization_approved=True,user_authorization=auth['user_reply']);p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);print('T586_REGISTERED_NOT_LAUNCHED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
