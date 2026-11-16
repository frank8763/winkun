"""Register a fixed four-prefix recovery qualification; zero navigation learning."""
from pathlib import Path
import ast,copy,datetime,importlib.util,json,sys,time,io
import t583_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
 old=ROOT/'stage/runs/t582_fresh_multitask_pair';assert read(old/'audit.json')['state']=='PASS'
 assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T582.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids'])
 p=copy.deepcopy(read(old/'protocol.json'));rows=R.G.rows(old);indices=[0,12,16,20];source=[rows[i] for i in indices]
 jobs=[r['request']['job'] for r in source];refs=[dict(path=str(ROOT/'stage/runs/t581_preserved_t550_rl/episodes/full_0.npz'))]+[next(r for r in p['keep_rows'] if r['job_id']==j['id']) for j in jobs[1:]]
 ks=[];rules=[]
 for row,ref in zip(source,refs):
  a=R.E.N.S.arrays(row['path']);b=R.E.N.S.arrays(ref['path']);n=min(len(a['poses']),len(b['poses']));d=a['poses'][:n]-b['poses'][:n]
  hit=np.flatnonzero((np.linalg.norm(d[:,:2],axis=1)>=.03)|(np.abs(np.arctan2(np.sin(d[:,2]),np.cos(d[:,2])))>=np.deg2rad(3.)))
  k=max(0,int(hit[0])-1) if len(hit) else len(a['states'])//2;assert k>0;ks.append(k)
  rules.append(dict(job=row['job_id'],k=k,source=row['path'],reference=ref['path'],reference_is_expert=False,rule='BEFORE_FIRST_3CM_OR_3DEG_OR_HALFWAY'))
 model=old/'initial.pt.gz';torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=L.frozen(p,model)
 assert q.phash()==source[0]['policy_hash'];initial=dict(path=str(model),sha256=sha(model),bytes=model.stat().st_size,policy_hash=q.phash())
 epoch=lambda t:datetime.datetime.fromisoformat('2026-11-16T'+t+':00+08:00').timestamp()
 cap=3*sum(j['case']['step_cap'] for j in jobs);assert cap==697008
 p.update(id='T583_FIXED_GAP_ACTUAL_NOISY_PREFIX_RECOVERY_QUALIFICATION',out=str(OUT),jobs=jobs,seeds=[r['seed'] for r in source],source_rows=source,teacher_firsts=ks,prefix_rules=rules,initial=initial,
  main_seconds=900,learning_seconds=0,audit_seconds=300,transport_seconds=300,source_seconds=180,
  absolute_learning_stop=epoch('12:45'),absolute_native_stop=epoch('12:45'),absolute_all_stop=epoch('13:00'),
  max_physics=cap,max_commands=16,max_bytes=16*2**20,child_bytes=16*2**20,inputs={},protected_files={},
  authorization_approved=True,user_authorization='2026-11-16 user: 下一步开始; existing13:00 all-stop retained; separate bounded fixed training-teacher qualification only',
  new_Actor_Adam_cap=0,shadow_Adam_cap=0,audit_discarded_Adam_cap=0,preflight_discarded_Adam_cap=0,new_RL=0,new_BC=0,new_Critic=0,
  scope='Four fixed historical stochastic sources rebuilt, four mixed GAP recoveries and four exact replays. No learning or autonomous promotion; qualified safe GOAL data only. No grid, retry, continuation or next launch.',
  count_contract='T583 operative fields only; inherited training fields loading compatibility only',keep_rows=[])
 upper=10*p['full_bytes']+2*2**20;assert upper<=p['max_bytes'];glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
 R.setup()
 # Existing full trajectory provides a real-input, zero-native Actor/history/density/reward audit.
 for row in source:
  a=R.E.N.S.arrays(row['path']);R.F.M.audit_full_history(p,row,a)
  n=len(a['states']);a=dict(a,teacher_actions=np.zeros((n,2),np.float32),teacher_scans=np.zeros((n,48),np.float32),teacher_vehicle_parameters=np.zeros(5),teacher_first_declared=np.asarray(1,np.int64))
  z=io.BytesIO();np.savez_compressed(z,**a);assert len(z.getvalue())<=p['full_bytes']
 spec=importlib.util.spec_from_file_location('_t583_transport_check',HERE/'transport.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);assert mod.OUT==OUT
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',native=0,optimizer=0,actual4_historical_full_density_history_reward=True,source_policy_hash=q.phash(),teacher_firsts=ks,
  stored_existing_full_plus_teacher_fields_fit=True,transport_OUT_matches=True,own_storage_upper=upper,global_worst=glob,
  no_new_native_engineering_probe=True,actual_mixed_native_contract_will_be_fail_closed_in_main=True))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 files=[*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',DOC/'AUTHORIZATION.json',Path(p['actor_configuration']),old/'audit.json',old/'SEALED.json',model]+[Path(r['path']) for r in source+refs]
 for f in files:p['inputs'][str(f)]=sha(f)
 m=p['base_model']
 while m:
  assert sha(m['path'])==m['sha256'];p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
 p['protected_files'][str(model)]=sha(model)
 for j in jobs:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(max_physics=cap,max_commands=16,main_seconds=900,audit_seconds=300,transport_seconds=300,source_seconds=180,all_work_reserve_seconds=1710,own_bytes=p['max_bytes'],own_storage_upper=upper,global_worst=glob,new_Adam=0,new_native_cases=8,exact_replays=4,historical_repeats=4,all_stop='2026-11-16 13:00+08:00'))
 print('T583_PREFLIGHT_PASS',ks,glob,flush=True)

def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');assert time.time()+1710<p['absolute_all_stop'] and time.time()+900<p['absolute_native_stop']
 assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True)
 print('T583_REGISTERED_NOT_LAUNCHED',flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
