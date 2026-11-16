"""One new bounded learning pilot; no extension of any ended run."""
from pathlib import Path
import ast,copy,datetime,importlib.util,json,sys,time
import t584_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
 old=ROOT/'stage/runs/t583_qualified_gap_recovery';assert read(old/'audit.json')['state']=='PASS'
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T583.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids'])
 p=copy.deepcopy(read(ROOT/'stage/runs/t582_fresh_multitask_pair/protocol.json'));prior=read(old/'protocol.json');rec=R.G.rows(old)[5:8]
 hand=[dict(j,source_path=r['path'],student_takeover_first=k) for j,r,k in zip(prior['jobs'][1:],prior['source_rows'][1:],prior['teacher_firsts'][1:])]
 epoch=lambda t:datetime.datetime.fromisoformat('2026-11-16T'+t+':00+08:00').timestamp()
 keys=[j for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']];cap=sum(j['case']['step_cap'] for j in p['evaluation_jobs']+keys)+4*sum(j['case']['step_cap'] for j in hand)
 p.update(id='T584_UNBOUNDED_RESIDUAL_QUALIFIED_RECOVERY_LEARNING',out=str(OUT),recovery_rows=rec,handoff_jobs=hand,
  BC_lr=.0003,BC_steps=100,keep_weight=100.,main_seconds=780,learning_seconds=240,audit_seconds=240,transport_seconds=180,source_seconds=90,
  absolute_learning_stop=epoch('12:42'),absolute_native_stop=epoch('12:50'),absolute_all_stop=epoch('13:00'),
  max_physics=cap,max_commands=69,max_bytes=20*2**20,child_bytes=20*2**20,inputs={},protected_files={},
  new_Actor_Adam_cap=100,shadow_Adam_cap=100,audit_discarded_Adam_cap=100,preflight_discarded_Adam_cap=12,new_RL=0,new_Critic=0,
  authorization_approved=True,user_authorization='2026-11-16 user: 下一轮开始; qualified recovery learnability and closed-loop preservation pilot; existing13:00 all-stop retained',
  scope='Same initial T550 function, remove .1 residual location cap, fixed100 supervised Adam+100shadow,27 history keeps;3 initial/3 learned physical-source-prefix handoffs,49 original-start autonomous,10exact. Not on-policy RL. No next branch/grid.',
  count_contract='T584 explicit fields only; inherited training keys loading compatibility only')
 upper=12*p['full_bytes']+45*p['compact_bytes']+3*p['model_bytes']+3*2**20;assert upper<=p['max_bytes'];glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
 R.setup();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);data=L.dataset(p);q=L.fresh(p);oldq=L.Old.fresh(p);bounds=[]
 for row in rec:
  a=R.E.N.S.arrays(row['path']);x=torch.tensor(a['states']);k=int(a['teacher_first_declared']);f,b=L.feature(q,x)
  assert torch.equal(q.sequence(x),oldq.sequence(x)) and torch.equal(L.loc(q,f,b),q.sequence(x))
  base=b[k:].detach().numpy();target=a['actions'][k:];gap=np.maximum(np.clip(base-.1,-1,1)-target,0)+np.maximum(target-np.clip(base+.1,-1,1),0)
  bounds.append(dict(job=row['job_id'],points=len(target),fraction_outside_old_bound=float(np.mean(np.any(gap>1e-6,axis=1))),unavoidable_old_command_max=gap.max(0).tolist(),unavoidable_old_command_MAE=gap.mean(0).tolist()))
 a,b=L.make(p),L.make(p);s=L.state(a);t=time.time();u=L.fit(a,data,p,lambda:None,lambda *_:None,steps=2);v=L.fit(b,data,p,lambda:None,lambda *_:None,steps=2);assert u==v and digest(L.state(a))==digest(L.state(b));estimate=(time.time()-t)*50
 mature=L.state(a);L.load(a,s);assert digest(L.state(a))==digest(s);L.load(a,mature);assert digest(L.state(a))==digest(mature)
 f=DOC/'discarded.pt.gz';model=L.export(a,p,f);assert L.frozen(p,f).phash()==a[0].phash();f.unlink()
 assert estimate<180,estimate
 spec=importlib.util.spec_from_file_location('_t584_transport',HERE/'transport.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert m.OUT==OUT
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE',new_native=0,discarded_synthetic_Adam=4,earlier_discarded_Adam=8,preflight_total_discarded_Adam=12,actual_recovery_arrays_used_only_discarded_preflight=True,old_bound=bounds,initial_function_on3_complete_histories_exact=True,history_features_and_sequence_exact=True,live_shadow_rollback_and_checkpoint_exact=True,estimated_live_shadow100_seconds=estimate,own_storage_upper=upper,global_worst=glob,transport_OUT_matches=True))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 files=[*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'AUTHORIZATION.json',Path(p['actor_configuration']),old/'audit.json',old/'SEALED.json']+[Path(r['path']) for r in p['keep_rows']+rec+prior['source_rows'][1:]]
 for f in files:p['inputs'][str(f)]=sha(f)
 m=p['base_model']
 while m:
  assert sha(m['path'])==m['sha256'];p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
 for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(max_physics=cap,max_commands=69,full10_compact45=True,handoff6_exact6_autonomous49_exact4=True,main_seconds=780,learning_seconds=240,audit_seconds=240,transport_seconds=180,source_seconds=90,all_work_reserve_seconds=1320,own_bytes=p['max_bytes'],own_storage_upper=upper,global_worst=glob,live_Adam=100,shadow_Adam=100,audit_historical_Adam=100,preflight_Adam=12,all_stop='2026-11-16 13:00+08:00'))
 print('T584_PREFLIGHT_PASS',cap,estimate,glob,flush=True)

def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');assert time.time()+1320<p['absolute_all_stop'] and time.time()+780<p['absolute_native_stop'] and time.time()+240<p['absolute_learning_stop']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);print('T584_REGISTERED_NOT_LAUNCHED',flush=True)
def finalize_gate():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');v=read(DOC/'PREFLIGHT.json');assert v['state']=='PASS_ZERO_NATIVE' and v['preflight_total_discarded_Adam']==12
 p['initial_handoff_learning_gate']='If all3 initial deterministic handoffs safeGOAL, stop before learning,49evaluation null;10commands;0navigationAdam. Otherwise original100-step pilot.'
 v['final_gate_zero_native_zero_optimizer_source_review']=True;save(DOC/'PREFLIGHT.json',v)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json']:
  if f.suffix=='.py':ast.parse(f.read_text())
  p['inputs'][str(f)]=sha(f)
 save(DOC/'PROPOSAL.json',p);print('T584_FINAL_GATE_ZERO_NATIVE_ZERO_ADAM',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
