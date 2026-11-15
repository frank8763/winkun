"""Complete the same frozen student's consumed development baseline, not qualify it."""
from pathlib import Path
import sys,json,time,shutil
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import t569_runtime as R
ROOT,OUT,DOC,L,torch,np,read,save,sha,digest=R.ROOT,R.OUT,R.DOC,R.L,R.torch,R.np,R.read,R.save,R.sha,R.digest
OLD=ROOT/'stage/runs/t567r_single_worker_recovery'
def build():
 p=read(ROOT/'stage/runs/t566_constrained_joint_step/protocol.json');assert read(OLD/'audit.json')['state']=='PASS';imported=R.G.rows(OLD);assert len(imported)==3;ids={r['job_id'] for r in imported};model=read(OLD/'protocol.json')['model'];jobs=[j for j in p['evaluation_jobs'] if j['id'] not in ids];assert len(jobs)==29
 p.update(id='T569_FROZEN_T562_MISSING29_COMPLETE32_BASELINE',jobs=jobs,model=model,models=dict(student=model),imported_rows=imported,prior23_success_ids=p['protected_success_ids'],max_physics=sum(j['case']['step_cap'] for j in jobs),max_commands=33,main_seconds=900,max_main_seconds=900,learning_seconds=0,audit_seconds=180,transport_seconds=180,source_seconds=120,max_bytes=3*2**20,child_bytes=3*2**20,full_bytes=1280*1024,compact_bytes=12*1024,scope='FreezeT562modelmeasuremissing29oncecompact;importsamepolicy3alreadyfull/exact fromT567R,toformsame-model32consumeddevelopmentbaseline.0newoptimizer/RL/teacher;notpassofflinegateorqualification.29nativehasheswithout29independentphysicalreplays;fullRAMhistorycheckinmainnotindependentofflinefullaudit. Noautomaticlearning.')
 return p

def preflight():
 assert not OUT.exists() and not (DOC/'PREFLIGHT.json').exists();DOC.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);p=build();cfg=read(p['actor_configuration']);cfg.update(p);q=L.frozen(cfg,p['model']['path']);assert q.phash()==p['model']['policy_hash'];assert all(r['policy_hash']==q.phash() and r['loaded_model_sha256']==p['model']['sha256'] for r in p['imported_rows']);a=R.E.N.S.arrays(p['imported_rows'][0]['path']);assert sha(p['imported_rows'][0]['path'])==p['imported_rows'][0]['sha256'];x=torch.tensor(a['states'][:16]);q.sample=False;q.reset_noise();h=None
 with torch.no_grad():
  for i,z in enumerate(x):u,loc,h,_=q.step(z[None],h);assert np.array_equal(u[0].numpy(),a['actions'][i])
 mx=max((j['case']['step_cap']+99)//100 for j in p['jobs']);rng=np.random.default_rng(569);test=dict(actions=rng.uniform(-1,1,(mx,2)).astype(np.float32),native_counts=rng.integers(0,256,mx,np.uint8),end_flags=rng.integers(0,256,(mx,3),np.uint8));compact=len(R.Base.pack(test));assert compact<=p['compact_bytes'];packed=R.E.N.S.packed(a);n=len(a['states']);bound=65536
 for v in packed.values():bound+=mx*int(v[0].nbytes) if v.ndim and v.shape[0]==n else (mx+1)*int(v[0].nbytes) if v.ndim and v.shape[0]==n+1 else v.nbytes
 assert bound<=p['full_bytes'];storage=p['full_bytes']+30*p['compact_bytes']+int(1.25*2**20);assert storage<=p['max_bytes'];used=R.used(ROOT/'stage/runs');assert used+p['max_bytes']+131072<2**30;assert 'range(len(self.children))' in (HERE/'t569_runtime.py').read_text();save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',epoch=time.time(),new_native=0,new_optimizer=0,source={str(f):sha(f) for f in HERE.glob('*.py')},max_decisions=mx,compact_incompressible_bytes=compact,partial_full_uncompressed_upper=bound,storage_bytes=storage,global_worst=used+p['max_bytes']+131072,same_model_import3_hash_exact=True,no_qualification_or_fullphysicalrepeat_claim=True));print('T569_ZERO_PREFLIGHT_PASS',p['max_physics'],storage)

def prepare():
 assert not OUT.exists();pref=read(DOC/'PREFLIGHT.json');assert pref['state']=='PASS_ZERO_NATIVE_ZERO_OPTIMIZER' and all(sha(f)==h for f,h in pref['source'].items());p=build();a=read(DOC/'AUTHORIZATION.json');assert a['scope_approved'];p.update(absolute_all_stop=a['all_stop_epoch'],absolute_native_stop=a['native_stop_epoch'],absolute_learning_stop=a['learning_stop_epoch'])
 prior=[read(ROOT/'stage/docs/day_20261115/t555/FINAL_REPORT.json')['ledger']]
 for n in ['t557_bc_stage_isolation','t559_smooth_policy_consolidation','t561_sine_policy_consolidation','t562_full_history_representation','t563_guarded_tail_fit','t564_local_peak_descent','t565_joint_residual_step','t566_constrained_joint_step','t567_frozen_closed_loop_diagnosis','t567r_single_worker_recovery','t568_student_recovery_qualification']:prior.append(read(ROOT/'stage/runs'/n/'summary.json'))
 phys=sum(x['physics'] for x in prior);cmd=sum(x['commands'] for x in prior);main=sum(x.get('main_seconds',x.get('elapsed_s',0)) for x in prior);assert phys+p['max_physics']<=19_000_000 and cmd+33<=600 and main+900<=12300
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'AUTHORIZATION.json',DOC/'PREFLIGHT.json',DOC/'SOURCES.md',Path(p['model']['path']),OLD/'cases.jsonl',OLD/'audit.json']+[Path(r['path']) for r in p['imported_rows']]:p['inputs'][str(f)]=sha(f)
 p['protected_files'][p['model']['path']]=p['model']['sha256'];assert R.used(ROOT/'stage/runs')+p['max_bytes']+131072<2**30 and time.time()+900+180+180+120+60<p['absolute_all_stop'];assert time.time()+900<p['absolute_native_stop'];OUT.mkdir();blob=json.dumps(p,separators=(',',':'),allow_nan=False)+'\n';(HERE/'protocol.json').write_text(blob);(OUT/'protocol.json').write_text(blob);R.verify(p,True);save(DOC/'BUDGET.json',dict(state='REGISTERED_NOT_LAUNCHED',physics=p['max_physics'],commands=33,main_seconds=900,day_physics_worst=phys+p['max_physics'],day_commands_worst=cmd+33,day_main_worst=main+900,own_bytes=p['max_bytes'],global_worst=R.used(ROOT/'stage/runs')-R.used(OUT)+p['max_bytes']+131072,new_optimizer=0,new_RL=0,new_native29=29,imported_same_model3=3,new_independent_native_replays=0));print('T569_REGISTERED')
if __name__=='__main__':globals()[sys.argv[1]]()
