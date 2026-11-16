"""Prepare without optimizers/native work; only explicit new-budget authorization admits launch."""
from pathlib import Path
import ast,copy,datetime,json,sys,time,importlib.util
import t590_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(exist_ok=True,parents=True)
 source=ROOT/'stage/runs/t589_complete_success_history';assert read(source/'audit.json')['state']=='PASS';assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T589.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 assert all(not Path('/proc',str(i)).exists() for i in read(source/'SEALED.json')['scientific_pids'])
 p=copy.deepcopy(read(ROOT/'stage/runs/t586_function_preserving_initialization/protocol.json'));coverage=read(source/'history_coverage.json');p['keep_rows']+=coverage['new_qualified_rows'];assert len(p['keep_rows'])==39 and len({r['job_id'] for r in p['keep_rows']})==39
 epoch=lambda x:datetime.datetime.fromisoformat('2026-11-16T'+x+':00+08:00').timestamp()
 keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];cap=2*(sum(j['case']['step_cap'] for j in p['evaluation_jobs']+keys)+2*sum(j['case']['step_cap'] for j in p['handoff_jobs']))
 p.update(id='T590_SAME39_HISTORY_FIXED_PENALTY_VS_PER_HISTORY_CONSTRAINTS',out=str(OUT),methods=['fixed_penalty','per_history_constraints'],BC_steps=512,BC_lr=.0003,keep_weight=100.,dual_rho=.1,keep_max_tolerance=.01,teacher_relative_gate=.8,
  main_seconds=1800,learning_seconds=900,audit_seconds=600,transport_seconds=180,source_seconds=90,absolute_learning_stop=epoch('14:40'),absolute_native_stop=epoch('14:50'),absolute_all_stop=epoch('15:00'),
  max_physics=cap,max_commands=126,max_bytes=20*2**20,child_bytes=20*2**20,model_bytes=512*1024,inputs={},protected_files={},authorization_approved=False,
  new_Actor_Adam_cap=1024,shadow_Adam_cap=1024,audit_discarded_Adam_cap=1024,preflight_discarded_Adam_cap=8,new_RL=0,new_Critic=0,
  user_authorization='Preparation requested; new scientific allocation/deadline pending. T58914:30 approval was data-only and is not silently reused.',
  scope='Two same-data same-initial-function512BC arms. Old39maxlocation<=.01 and all3teacherMAE<=80%initial gate. Conditional3nativehandoff+3exact, if any new safe then49original+4compactexact; keep39IDs all preserved. No407/fresh/reload/globalpromotion. No seed/duration/weight grid or next mechanism.')
 upper=8*p['full_bytes']+92*p['compact_bytes']+4*p['model_bytes']+3*2**20;glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert upper<=p['max_bytes'] and glob<2**30
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);R.setup(DOC);data=L.dataset(p);q=L.fresh(p);reference=L.Prior.Prior.fresh(p);initial=digest(q.state_dict());checks=0
 for row in p['keep_rows']+p['recovery_rows']:
  a=R.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);f,b=L.feature(q,x)
  with torch.no_grad():assert torch.equal(L.loc(q,f,b),b) and torch.equal(q.sequence(x),reference.sequence(x))
  checks+=1
 assert max(L.metrics(q,data)['keep_max_by_history'])==0 and digest(q.state_dict())==initial
 # The PHR inequality penalty is inactive at lambda0 for a strict feasible point,
 # positive with an outward gradient when violated, and duals never negative.
 g=torch.tensor([-1.,0.,1.],dtype=torch.float64,requires_grad=True);rho=p['dual_rho'];pen=torch.relu(rho*g).square()/(2*rho);grad=torch.autograd.grad(pen.sum(),g)[0]
 assert torch.equal(grad,torch.tensor([0.,0.,rho],dtype=torch.float64)) and torch.equal(torch.relu(rho*g.detach()),torch.tensor([0.,0.,rho],dtype=torch.float64))
 assert not torch.cuda.is_initialized()
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 spec=importlib.util.spec_from_file_location('_t590_transport',HERE/'transport.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);assert t.OUT==OUT
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_PREPARED_ONLY',full_histories_initial_function_exact=checks,keep_unique_IDs=39,qualified_full_teacher_suffixes=3,initial_all39_max_location_error=0,new_native=0,optimizer_objects=0,optimizer_updates=0,synthetic_constraint_gradient_checks=1,model_SHA_unchanged=True,own_upper_bytes=upper,global_worst_bytes=glob))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),source/'audit.json',source/'history_coverage.json']+[Path(r['path']) for r in p['keep_rows']+p['recovery_rows']]:p['inputs'][str(f)]=sha(f)
 for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 for j in p['handoff_jobs']:p['inputs'][j['source_path']]=sha(j['source_path'])
 model=p['base_model']
 while model:p['inputs'][model['path']]=model['sha256'];p['protected_files'][model['path']]=model['sha256'];model=model.get('parent')
 save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(state='PREPARED_NOT_REGISTERED_NOT_LAUNCHED',max_physics=cap,max_commands=126,own_bytes=p['max_bytes'],own_upper_bytes=upper,global_worst=glob,main_seconds=1800,learning_seconds=900,audit_seconds=600,transport_seconds=180,source_seconds=90,total_admission_seconds=2700,live_Adam=1024,shadow_Adam=1024,audit_discarded_Adam=1024,preflight_discarded_Adam=8,all_stop_proposed='15:00',new_budget_authorization='PENDING'))
 print('T590_ZERO_NATIVE_PREFLIGHT_PASS_PREPARATION_ONLY',cap,glob,flush=True)

def optimizer_preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT_OPTIMIZER.json').exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json');assert auth['approved'] is True and auth['all_stop_epoch']==p['absolute_all_stop'] and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands']
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.dataset(p);total=0;states=[]
 for method in p['methods']:
  pc=dict(p,method=method);a,b=L.make(pc),L.make(pc);before=L.state(a);states.append(digest(before));u=L.fit(a,data,pc,lambda:None,lambda *_:None,steps=2);v=L.fit(b,data,pc,lambda:None,lambda *_:None,steps=2);total+=4
  assert u==v and digest(L.state(a))==digest(L.state(b));mature=L.state(a);L.load(a,before);assert digest(L.state(a))==digest(before);L.load(a,mature);assert digest(L.state(a))==digest(mature)
  f=DOC/'discarded_mature.pt.gz';meta=L.export(a,pc,f);assert L.frozen(pc,f).phash()==a[0].phash();f.unlink()
 assert states[0]==states[1] and total==8
 save(DOC/'PREFLIGHT_OPTIMIZER.json',dict(state='PASS_AUTHORIZED_ZERO_NATIVE',discarded_Adam=8,exact_shadow_ActorAdam_duals_mature_rollback_readback=True,new_native=0));p['inputs'][str(DOC/'PREFLIGHT_OPTIMIZER.json')]=sha(DOC/'PREFLIGHT_OPTIMIZER.json');save(DOC/'PROPOSAL.json',p)

def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');auth=read(DOC/'AUTHORIZATION.json');assert auth['approved'] is True and auth['all_stop_epoch']==p['absolute_all_stop'] and auth['max_physics']==p['max_physics'] and auth['max_commands']==p['max_commands'] and auth['own_bytes']==p['max_bytes'];assert read(DOC/'PREFLIGHT_OPTIMIZER.json')['state']=='PASS_AUTHORIZED_ZERO_NATIVE'
 assert time.time()+2700<p['absolute_all_stop'] and time.time()+1800<p['absolute_native_stop'] and time.time()+900<p['absolute_learning_stop'] and R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 p.update(authorization_approved=True,user_authorization=auth['user_reply']);p['inputs'][str(DOC/'AUTHORIZATION.json')]=sha(DOC/'AUTHORIZATION.json');OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);print('T590_REGISTERED_NOT_LAUNCHED',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
