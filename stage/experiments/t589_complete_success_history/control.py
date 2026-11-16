"""Complete missing full histories for12known-success development tasks, no learning."""
from pathlib import Path
import ast,copy,sys,time,json,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t589_complete_success_history';DOC=ROOT/'stage/docs/day_20261116/t589'
sys.path.insert(0,str(ROOT/'stage/experiments/t587_verified_short_correction'));import t587_runtime as R
L=R.L;read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def setup():R.OUT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup()
def prepare():
 assert not OUT.exists();p=copy.deepcopy(read(ROOT/'stage/runs/t587_verified_short_correction/protocol.json'));assert read(ROOT/'stage/runs/t588_actual_support_audit/audit.json')['state']=='PASS';ids=set(p['initial_success_ids']);covered={r['job_id'] for r in p['keep_rows']};jobs=[j for j in p['evaluation_jobs'] if j['id'] in ids-covered];assert len(jobs)==12
 cap=2*sum(j['case']['step_cap'] for j in jobs);assert cap==896008
 p.update(id='T589_MISSING12_SUCCESS_FULL_HISTORY_CAPTURE',out=str(OUT),capture_jobs=jobs,main_seconds=480,learning_seconds=0,audit_seconds=120,transport_seconds=120,source_seconds=90,
  absolute_native_stop=1794810120.,absolute_all_stop=1794810600.,max_physics=cap,max_commands=28,max_bytes=20*2**20,child_bytes=20*2**20,
  inputs={},protected_files={},authorization_approved=True,user_authorization='2026-11-16 user: 下一步做什么，继续; missing known-success history data completion, no learning; user explicitly authorized14:30 all-stop after full registration preparation',new_Actor_Adam_cap=0,shadow_Adam_cap=0,audit_discarded_Adam_cap=0,new_RL=0,new_Critic=0,scope='12 fixed known-success tasks absent from27history set;12full+12exact;0learning0newability claim. Preserve all failed baseline outcomes, no next launch.')
 upper=14*p['full_bytes']+2*2**20;assert upper<=p['max_bytes'];glob=R.used(ROOT/'stage/runs')+p['max_bytes']+131072;assert glob<2**30 and time.time()+840<p['absolute_all_stop'] and time.time()+480<p['absolute_native_stop']
 setup();torch.set_num_threads(1);q=L.frozen(p,p['initial']['path']);assert q.phash()==p['initial']['policy_hash'];assert not torch.cuda.is_initialized()
 for f in HERE.glob('*.py'):ast.parse(f.read_text())
 save(DOC/'PREFLIGHT.json',dict(state='PASS_ZERO_NATIVE_ZERO_OPTIMIZER',new_native=0,new_optimizer=0,missing12_known_IDs=[j['id'] for j in jobs],immutable_policy_hash=q.phash(),full14_slots_include2_worker_partial=True,own_storage_upper=upper,global_worst=glob))
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(ROOT)) and str(f).endswith('.py'):p['inputs'][str(Path(f).resolve())]=sha(f)
 for f in [*HERE.glob('*.py'),DOC/'PLAN.md',DOC/'PREFLIGHT.json',Path(p['actor_configuration']),ROOT/'stage/runs/t588_actual_support_audit/audit.json',Path(p['initial']['path'])]+[Path(r['path']) for r in p['keep_rows']]:p['inputs'][str(f)]=sha(f)
 p['protected_files'][p['initial']['path']]=p['initial']['sha256'];m=p['base_model']
 while m:p['inputs'][m['path']]=m['sha256'];p['protected_files'][m['path']]=m['sha256'];m=m.get('parent')
 for j in jobs:p['inputs'][j['exact_snapshot']]=sha(j['exact_snapshot'])
 save(DOC/'BUDGET.json',dict(max_physics=cap,max_commands=28,own_bytes=p['max_bytes'],own_storage_upper=upper,global_worst=glob,main_seconds=480,audit_seconds=120,transport_seconds=120,source_seconds=90,reserve_seconds=840,new_Adam=0,all_stop='14:30',native_stop='14:22'))
 OUT.mkdir();save(OUT/'protocol.json',p);save(HERE/'protocol.json',p);R.verify(p,True);print('T589_REGISTERED_NOT_LAUNCHED',glob,flush=True)

def main():
 p=read(OUT/'protocol.json');setup();R.verify(p,True);R.E.N.offline();R.E.L.install_stop_handlers();rt=R.Base.Runtime(p);state='FAILED';err=None;result=None
 try:
  rt.pool=R.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard);vv=rt.request(p['capture_jobs'],p['initial'],'KNOWN_SUCCESS_MISSING_FULL_HISTORY',full=True)
  for a,r in vv:R.F.M.audit_full_history(p,r,a)
  rr=[r for a,r in vv];rt.request(p['capture_jobs'],p['initial'],'EXACT_KNOWN_SUCCESS_FULL_HISTORY',full=True,reference=rr);safe=[r for r in rr if R.safe(r)];allpass=len(safe)==12
  coverage={r['job_id'] for r in p['keep_rows']}|{r['job_id'] for r in safe};save(OUT/'history_coverage.json',dict(existing27=p['keep_rows'],new_current_policy_rows=rr,new_qualified_rows=safe,known39_success_ids=p['initial_success_ids'],covered_known_success_ids=sorted(coverage&set(p['initial_success_ids'])),existing_rows_NOT_all_current_policy_native_samples=True))
  result=dict(current_policy12=R.G.R.M.tally(rr),all12_known_baselines_reproduced=allpass,covered_known_successes=len(coverage&set(p['initial_success_ids'])),new_Actor=0,new_Adam=0,new_RL=0,new_independent_success_tasks=0,full407=None,fresh16=None,reload78=None,globally_promoted=False);state='COMPLETE'
 except BaseException:err=traceback.format_exc();save(OUT/'failure.json',dict(error=err,no_restart=True,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=__import__('os').getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=err,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',err

def audit():
 p=read(OUT/'protocol.json');setup();R.verify(p,True);R.G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=R.G.rows(OUT);reps=R.G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==12
 for r in rows:
  a=R.E.N.S.arrays(r['path']);assert sha(r['path'])==r['sha256'] and int(a['native_counts'].sum())==r['physics'] and all(digest(w)==r['array_hashes'][k] for k,w in a.items());R.F.M.audit_full_history(p,r,a)
 by={r['path']:r for r in rows}
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert R.G.R.M.tally(rows)==s['result']['current_policy12'];assert s['commands']==28 and s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'];assert not (OUT/'optimizer_calls.jsonl').exists()
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_optimizer=0,all12_full_history_density_reward_and_native_exact=True,all_models_SHA_unchanged=True,physics=s['physics'],commands=s['commands']));print('T589_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
