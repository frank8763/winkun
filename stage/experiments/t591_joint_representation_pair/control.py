"""Prepare the fixed study without simulation or optimizer updates.

Authorization/registration are deliberately separate from a read-only preflight.
"""
from pathlib import Path
import ast,copy,importlib.util,json,sys,time
import t591_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists()
 DOC.mkdir(parents=True,exist_ok=True)
 previous=ROOT/'stage/runs/t590_per_history_constraints'
 assert read(previous/'audit.json')['state']=='PASS'
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T590.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 assert all(not Path('/proc',str(i)).exists() for i in read(previous/'SEALED.json')['scientific_pids'])
 p=copy.deepcopy(read(previous/'protocol.json'))
 model=ROOT/'stage/runs/t562_full_history_representation/proposal.pt.gz'
 assert sha(model)=='e3a20a54155b015cd2f922b5c9366cdbfff5b51b255cc0d6b9ab3b1b264f4ac0'
 cap=p['max_physics']+2*sum(j['case']['step_cap'] for j in p['handoff_jobs'])
 p.update(id='T591_MATCHED_SMOOTH_STUDENT_JOINT_ACTION_IMITATION',out=str(OUT),
  methods=['frozen_representation','trainable_representation'],student_initial=dict(path=str(model),sha256=sha(model)),student_seed=559,
  fit_steps=600,fit_lr=.0003,main_seconds=2400,learning_seconds=1500,audit_seconds=900,transport_seconds=300,source_seconds=120,
  absolute_learning_stop=None,absolute_native_stop=None,absolute_all_stop=None,
  max_physics=cap,max_commands=136,max_bytes=22*2**20,child_bytes=22*2**20,model_bytes=3*2**20,initial_model_bytes=1280*1024,
  inputs={},protected_files={},authorization_approved=False,new_Actor_Adam_cap=1200,shadow_Adam_cap=1200,audit_discarded_Adam_cap=1200,preflight_discarded_Adam_cap=8,new_RL=0,new_Critic=0,
  user_authorization='Preparation only; new bounded learning allocation and window pending. T59015:00 authorization not extended.',
  scope='Same T562 research initial and39old+3complete recovery histories.600matchedAdam each, different trainable parameter counts, not matched FLOPs. Boundary-aware recorded-action loss, equal old/recovery groups and equal trajectories within groups, half mean/half worst10percent. Both candidates get mandatory autonomous3handoff+exact irrespective of descriptive fit. Any safe handoff triggers49original-start+4keyexact. Protect39T550IDs and add a successful handoff or original-start ID to retain research candidate. No407/fresh/reload/globalpromotion, no RL or runtime teacher.')
 # Remove stale scientific fields instead of suggesting the old gates still apply.
 for key in ['BC_steps','BC_lr','keep_weight','dual_rho','keep_max_tolerance','teacher_relative_gate']:
  p.pop(key,None)
 upper=8*p['full_bytes']+101*p['compact_bytes']+2*p['model_bytes']+p['initial_model_bytes']+3*2**20
 glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072
 assert upper<=p['max_bytes'] and glob<2**30
 assert __import__('shutil').disk_usage(ROOT).free>=40*2**30
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.setup(DOC)
 data=L.dataset(p);q=L.initial(p);unchanged=digest(q.state_dict());packed=torch.nn.utils.rnn.pad_sequence([g['x'] for g in data])
 comparison=[];gradient=[]
 with torch.no_grad():
  for g in data:
   a=L.locations(q,g,p['methods'][0]);b=L.locations(q,g,p['methods'][1]);err=float((a-b).abs().max());assert err<1e-12
   comparison.append(dict(id=g['id'],kind=g['kind'],length=len(g['x']),labels=int(g['mask'].sum()),initial_forward_max_error=err))
  # Same input history and reset semantics as the deployment worker; true carry.
  for g in [data[0],data[-1]]:
   x=g['x'][:48];n=q.normalized(x);full,_=q.core(q.encoder(n)[:,None]);h=None;states=[]
   for row in n:
    y,h=q.core(q.encoder(row)[None,None],h);states.append(y[0,0])
   assert float((torch.stack(states)-full[:,0]).abs().max())<1e-12
 for method in p['methods']:
  q.requires_grad_(False)
  for v in L.variables(q,method):v.requires_grad_(True)
  q.zero_grad(set_to_none=True);start=time.monotonic();loss,parts=L.objective(q,data,method,packed);loss.backward();elapsed=time.monotonic()-start
  pars=L.variables(q,method);assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in pars)
  assert digest(q.state_dict())==unchanged
  gradient.append(dict(method=method,loss=float(loss.detach()),parts=parts,trainable_parameters=sum(v.numel() for v in pars),nonzero_gradient_values=sum(int(torch.count_nonzero(v.grad)) for v in pars),full_batch_forward_backward_seconds=elapsed))
 q.zero_grad(set_to_none=True)
 # Initial checkpoint has no Adam; reserve separately from the two trained models.
 file=DOC/'discarded_initial.pt.gz';meta=L.export((q,None),dict(p,model_bytes=p['initial_model_bytes']),file)
 assert L.frozen(p,file).phash()==q.phash();file.unlink()
 assert not torch.cuda.is_initialized()
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 spec=importlib.util.spec_from_file_location('_t591_transport',HERE/'transport.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);assert t.OUT==OUT
 check=dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_PREPARED_ONLY',new_native=0,optimizer_objects=0,optimizer_updates=0,full_batch_gradient_calls=2,full_histories=comparison,gradient_checks=gradient,initial_policy_hash=q.phash(),initial_bytes=meta['bytes'],model_SHA_unchanged=True,old_unique_IDs=39,qualified_complete_recovery_suffixes=3,own_upper_bytes=upper,global_worst_bytes=glob,initial_metrics=L.measure(q,data,p['methods'][0]))
 save(DOC/'PREFLIGHT.json',check)
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),previous/'audit.json',model]+[Path(r['path']) for r in p['keep_rows']+p['recovery_rows']]:p['inputs'][str(f)]=sha(f)
 for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 for j in p['handoff_jobs']:p['inputs'][j['source_path']]=sha(j['source_path'])
 ancestor=p['base_model']
 while ancestor:p['inputs'][ancestor['path']]=ancestor['sha256'];p['protected_files'][ancestor['path']]=ancestor['sha256'];ancestor=ancestor.get('parent')
 p['protected_files'][str(model)]=sha(model)
 save(DOC/'PROPOSAL.json',p)
 save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=cap,max_commands=136,own_bytes=p['max_bytes'],own_upper_bytes=upper,global_worst_bytes=glob,main_seconds=2400,learning_seconds=1500,audit_seconds=900,transport_seconds=300,source_seconds=120,total_admission_seconds=3750,requested_window_seconds=5400,live_Adam=1200,shadow_Adam=1200,audit_discarded_Adam=1200,preflight_discarded_Adam=8,new_budget_authorization='PENDING'))
 print('T591_ZERO_NATIVE_ZERO_OPTIMIZER_PREFLIGHT_PASS',cap,glob,json.dumps(gradient),flush=True)

def optimizer_preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists()
 p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json')
 assert auth['approved'] is True and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands'] and time.time()<auth['all_stop_epoch']
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.dataset(p);total=0;initials=[]
 for method in p['methods']:
  pc=dict(p,method=method);a,b=L.make(pc),L.make(pc);before=L.state(a);initials.append(a[0].phash())
  u=L.fit(a,data,pc,lambda:None,lambda *_:None,steps=2);v=L.fit(b,data,pc,lambda:None,lambda *_:None,steps=2);total+=4
  assert u==v and digest(L.state(a))==digest(L.state(b));mature=L.state(a);L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
  f=DOC/'discarded_mature.pt.gz';L.export(a,pc,f);assert L.frozen(pc,f).phash()==a[0].phash();f.unlink()
 assert initials[0]==initials[1] and total==8
 save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',discarded_Adam=8,exact_shadow_ActorAdam_mature_rollback_readback=True,new_native=0))
 p['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',p)

def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json')
 assert auth['approved'] is True and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands'] and auth['own_bytes']==p['max_bytes']
 assert read(DOC/'PREFLIGHT_OPTIMIZER.json')['state']=='PASS_AUTHORIZED_ZERO_NATIVE'
 for key in ['new_Actor_Adam_cap','shadow_Adam_cap','audit_discarded_Adam_cap','preflight_discarded_Adam_cap']:assert auth[key]==p[key]
 p.update(absolute_learning_stop=auth['learning_stop_epoch'],absolute_native_stop=auth['native_stop_epoch'],absolute_all_stop=auth['all_stop_epoch'])
 p.update(relative_window_seconds=auth['relative_window_seconds'],latest_launch_epoch=auth['latest_launch_epoch'])
 assert p['relative_window_seconds']==5400 and time.time()<p['latest_launch_epoch']
 assert time.time()+3750<p['absolute_all_stop'] and time.time()+2400<p['absolute_native_stop'] and time.time()+1500<p['absolute_learning_stop']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 p.update(authorization_approved=True,user_authorization=auth['user_reply']);p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir()
 blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);print('T591_REGISTERED_NOT_LAUNCHED',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
