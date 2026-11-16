"""Fixed4s hand-back qualification before learning only actually safe corrections."""
from pathlib import Path
import ast,copy,importlib.util,json,sys,time
import t587_runtime as R
ROOT,HERE,OUT,DOC,L=R.ROOT,R.HERE,R.OUT,R.DOC,R.L
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def prepare():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True)
 old=ROOT/'stage/runs/t586_function_preserving_initialization';assert read(old/'audit.json')['state']=='PASS';assert all(not Path('/proc',str(i)).exists() for i in read(old/'SEALED.json')['scientific_pids']);assert read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T586.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
 base=ROOT/'stage/runs/t584_unbounded_recovery_learning';p=copy.deepcopy(read(base/'protocol.json'));v=read(base/'summary.json')['result'];source=read(ROOT/'stage/runs/t583_qualified_gap_recovery/protocol.json');jobs=[dict(j,source_path=r['path'],gap_recovery_first=k,correction_decisions=20) for j,r,k in zip(source['jobs'][1:],source['source_rows'][1:],source['teacher_firsts'][1:])]
 p.update(id='T587_FIXED4S_VERIFIED_INTERVENTION_TARGET',out=str(OUT),initial=v['initial'],qualification_jobs=jobs,qualification_seeds=source['seeds'][1:],correction_decisions=20,historical_handoff_baseline=v['baseline_handoff'],
  main_seconds=900,learning_seconds=240,audit_seconds=240,transport_seconds=180,source_seconds=90,
  absolute_learning_stop=1794807600.,absolute_native_stop=1794808200.,absolute_all_stop=1794808800.,
  max_physics=3205286,max_commands=69,max_bytes=22*2**20,child_bytes=22*2**20,model_bytes=512*1024,
  new_Actor_Adam_cap=100,shadow_Adam_cap=100,audit_discarded_Adam_cap=100,preflight_discarded_Adam_cap=0,
  inputs={},protected_files={},authorization_approved=True,user_authorization='2026-11-16 user: 下一步做什么，继续. New fixed short-correction qualification/learning target within14:00 window; not continuation of endedT586 initialization pilot.',
  scope='3fixed actual states, exactly4sGAP then original frozen student;3exact; only safeGOAL handbacks20teacherdecisions train100BC+100shadow,0newRL.3purestudent handoffs+3exact;only ifnewhandoff success49original+4exact. No duration/seed/weight/grid, no next mechanism.',count_contract='T587 explicit fields; inherited fields only loading compatibility.')
 cap=4*sum(j['case']['step_cap'] for j in jobs)+sum(j['case']['step_cap'] for j in p['evaluation_jobs'])+sum(j['case']['step_cap'] for j in p['evaluation_jobs'] if j['id'] in p['key_replay_ids']);assert cap==p['max_physics']
 upper=12*p['full_bytes']+45*p['compact_bytes']+3*p['model_bytes']+3*2**20;assert upper<=p['max_bytes'];glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30
 R.setup();torch.set_num_threads(1);torch.use_deterministic_algorithms(True);q=L.frozen(p,p['initial']['path']);assert q.phash()==L.make(p)[0].phash()
 # make() constructs a fresh empty Adam only, no step. Reuse exact tested fit/load/export implementation.
 assert L.fit is L.Prior.fit and L.load is L.Prior.load and L.export is L.Prior.export
 checks=[]
 for r,k in zip(p['recovery_rows'],source['teacher_firsts'][1:]):
  a=R.E.N.S.arrays(r['path']);n=len(a['states']);mask=(np.arange(n)>=k)&(np.arange(n)<k+20);assert mask.sum()==20 and np.array_equal(a['actions'][mask],a['teacher_actions'][mask]);f,b=L.feature(q,torch.tensor(a['states']));assert torch.equal(L.loc(q,f,b),q.sequence(torch.tensor(a['states'])));assert f[mask].shape==(20,184);checks.append(dict(job=r['job_id'],selected20_teacher_actions_exact=True,full_causal_history_features=True,NOT_new4s_handback_qualification=True))
 for r in v['baseline_handoff']:
  a=R.E.N.S.arrays(r['path']);R.actor_tape_audit(p,r,a)
 spec=importlib.util.spec_from_file_location('_t587_transport',HERE/'transport.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);assert m.OUT==OUT
 for f in HERE.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER_STEPS',new_native=0,optimizer_steps=0,one_empty_Adam_constructed_only=True,window_selection_contract=checks,initial3_handoff_complete_history_density_reward=True,optimizer_functions_exact_T584_alias=True,mature_ActorAdam_rollback_proof=str(base/'audit.json'),transport_OUT_matches=True,own_storage_upper=upper,global_worst=glob))
 for mod in list(sys.modules.values()):
  f=getattr(mod,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 files=[*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'SOURCES.md',DOC/'PREFLIGHT.json',DOC/'AUTHORIZATION.json',Path(p['actor_configuration']),old/'audit.json',old/'SEALED.json',base/'audit.json',base/'summary.json',Path(p['initial']['path'])]+[Path(r['path']) for r in p['keep_rows']+p['recovery_rows']+v['baseline_handoff']]
 for f in files:p['inputs'][str(f)]=sha(f)
 p['protected_files'][p['initial']['path']]=p['initial']['sha256'];m=p['base_model']
 while m:
  assert sha(m['path'])==m['sha256'];p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
 for j in jobs:p['inputs'][j['source_path']]=sha(j['source_path'])
 for j in p['evaluation_jobs']:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 save(DOC/'PROPOSAL.json',p);save(DOC/'BUDGET.json',dict(max_physics=cap,max_commands=69,own_bytes=p['max_bytes'],own_storage_upper=upper,global_worst=glob,main_seconds=900,learning_seconds=240,audit_seconds=240,transport_seconds=180,source_seconds=90,total_reserve_seconds=1440,conditional_full10_compact45=True,preflight_optimizer_steps=0,live_Adam_cap=100,shadow_Adam_cap=100,audit_historical_Adam_cap=100,new_RL=0,all_stop='2026-11-16 14:00+08:00'))
 print('T587_PREFLIGHT_PASS',cap,glob,flush=True)

def register():
 assert not OUT.exists();p=read(DOC/'PROPOSAL.json');assert time.time()+1440<p['absolute_all_stop'] and time.time()+900<p['absolute_native_stop'] and time.time()+240<p['absolute_learning_stop'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30
 for f,h in p['inputs'].items():assert sha(f)==h,f
 OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);print('T587_REGISTERED_NOT_LAUNCHED',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
