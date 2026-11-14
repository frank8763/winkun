"""Finite compatibility pilot; 13 autonomous cases and 13 exact replays only."""
from pathlib import Path
import fcntl,os,shutil,sys,time,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'stage/runs/t546_functional_residual';DOC=ROOT/'stage/docs/day_20261114'
sys.path.insert(0,str(HERE));import t546_actor as A
sys.path.insert(0,str(ROOT/'stage/experiments/t542_offline_field_relay'));import t542_runtime as Base
F,G,E,P,np,torch=Base.F,A.G,Base.E,A.P,A.np,A.torch
read,save,sha,digest=G.read,G.save,G.sha,G.digest

def used(folder):return sum(f.stat().st_size for f in Path(folder).rglob('*') if f.is_file())
def verify(p,full=False):
 root=read(OUT/'protocol.json');assert sha(HERE/'protocol.json')==sha(OUT/'protocol.json')
 assert time.time()<root['absolute_all_stop'] and used(OUT)<=root['max_bytes'] and used(ROOT/'stage/runs')-used(OUT)+root['max_bytes']<=2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<={14,15}
 for f,h in (root['inputs'] if full else root['protected_files']).items():assert sha(f)==h,f

def configuration(root):
 p=read(root['actor_configuration']);p.update(root);p['out']=str(OUT);return p
Base.OUT=OUT;Base.HERE=HERE;Base.DOC=DOC;Base.verify=verify

def setup():
 Base.setup(OUT);F.OUT=OUT;F.HERE=HERE;F.verify=verify;F.R.OUT=OUT;F.R.HERE=HERE;F.R.verify=verify;P.frozen=A.frozen
 torch.set_num_threads(1);torch.use_deterministic_algorithms(True)

def worker(pipe,index,p,counts):
 G.T.Session=F.FieldSession;G.B.native_setup=F.field_setup;P.frozen=A.frozen
 G.cpu_worker(pipe,index,p,counts)

class Pool(F.Pool):
 def __init__(self,p,counts,guard):
  self.p=p;self.guard,self.children,self.pipes=guard,[],[];ctx=G.R.M.mp.get_context('spawn')
  try:
   for i in range(2):
    a,b=ctx.Pipe();c=ctx.Process(target=worker,args=(b,i,p,counts));c.start();b.close();self.children.append(c);self.pipes.append(a)
   self.ready=[self.receive(i) for i in range(2)];assert all(v['state']=='READY' for v in self.ready)
   save(OUT/'worker_generations.json',[dict(parent=os.getpid(),workers=[c.pid for c in self.children],epoch=time.time(),CPU=[14,15],GPU=False)])
  except BaseException:self.close();raise

def teacher_physical_match(a,p):
 old=E.N.S.arrays(p['recovery_row']['path'])
 names=['states','next_states','actions','native_counts','poses','next_poses','end_flags','rewards','history_poses','history_goals','history_times','initial_native_integration','final_native_integration','hidden_before','last_hidden_after','task_goal_xy','scene_sha256']
 mismatched=[name for name in names if not np.array_equal(a[name],old[name])]
 return dict(exact=not mismatched,mismatched=mismatched,fields=names,policy_locations_logp_ownership_NOT_same_teacher=True)

def main():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);E.N.offline();E.L.install_stop_handlers();rt=Base.Runtime(p);state='FAILED';error=None
 result=dict(reason='NOT_EVALUATED',new_RL=0,Actor_Adam=0,Value=0,closed_form_live_solves=0,shadow_solves=0,qualification=None,model=None,panel=None,accepted=False,old8=None,field24=None,old275=None,six54=None,eight54=None,fresh16=None,reload78=None)
 try:
  rt.role='FIXED_FUNCTION_COMPATIBILITY';rt.guard(True)
  try:
   q,metric=A.fit(p,lambda step,v:save(OUT/'fit_progress.json',dict(step=step,metrics=v,epoch=time.time())));result['closed_form_live_solves']=1
   shadow,other=A.fit(p);result['shadow_solves']=1;assert metric==other and q.phash()==shadow.phash();rt.guard(True)
   model=A.export(q,p,OUT/'proposal.pt.gz');assert A.frozen(p,model['path']).phash()==q.phash();result.update(qualification=metric,model=model)
  except AssertionError as e:
   result.update(reason='FIXED_CONSTRUCTION_QUALIFICATION_REJECTED_ZERO_NATIVE',closed_form_live_solves=(read(OUT/'fit_progress.json').get('metrics',{}).get('closed_form_solves',0) if (OUT/'fit_progress.json').exists() else 0),qualification_failure=str(e),qualification_trace=traceback.format_exc())
   save(OUT/'qualification_rejection.json',dict(error=result['qualification_trace'],last_progress=read(OUT/'fit_progress.json') if (OUT/'fit_progress.json').exists() else None,scientific_no_retry=True))
  if result['model']:
   rt.pool=Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
   first=rt.request(p['panel_jobs'][:1],model,'PURE_ACTOR_OLD_LEFT_FULL',full=True);aa,rr=first[0];rt.request(p['panel_jobs'][:1],model,'EXACT_OLD_LEFT_FULL',full=True,reference=[rr]);physical=teacher_physical_match(aa,p);save(OUT/'recovery_physical_comparison.json',physical)
   # Failure of the full recovery stops this finite construction; no grids.
   if rr['success'] and not rr['collision'] and physical['exact']:
    rest=rt.request(p['panel_jobs'][1:],model,'PURE_ACTOR_ORIGINAL12_FIELD_SUCCESS');rt.request(p['panel_jobs'][1:],model,'EXACT_ORIGINAL12_FIELD_SUCCESS',reference=[r for a,r in rest])
    historical={r['job_id']:r for r in p['protected_rows']};matches=[]
    for a,r in rest:
     previous=historical[r['job_id']];exact=r['array_hashes']==previous['array_hashes'] and r['integration_sha256']==previous['integration_sha256'];matches.append(dict(id=r['job_id'],exact_parent_trajectory=exact,qualified=r['field_detail']['field_qualified_success']))
    result['panel']=dict(old_left=G.R.M.tally([rr]),known12=G.R.M.tally([r for a,r in rest]),parent_trajectory_matches=matches,all13_pass=all(x['exact_parent_trajectory'] and x['qualified'] for x in matches),protected_generalization_NOT_tested=True)
    result['reason']='FINITE_13_COMPATIBILITY_NATIVE_EXACT' if result['panel']['all13_pass'] else 'FINITE_CONSTRUCTION_NATIVE_FAILURE'
   else:
    result['panel']=dict(old_left=G.R.M.tally([rr]),teacher_physical_comparison=physical,known12=None,all13_pass=False);result['reason']='FINITE_RECOVERY_NATIVE_MISMATCH_STOP'
  state='COMPLETE'
 except BaseException:
  error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands if rt.pool else 0,elapsed_s=time.time()-rt.start,error=error,result=result));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=configuration(read(OUT/'protocol.json'));setup();verify(p,True);s=read(OUT/'summary.json');assert s['state']=='COMPLETE' and not (OUT/'audit.json').exists();G.stopped(OUT);r=s['result']
 if r['model']:
  q,m=A.fit(p);assert m==r['qualification'] and q.phash()==r['model']['policy_hash'] and digest(dict(actor=q.state_dict()))==r['model']['joint_hash'];assert sha(r['model']['path'])==r['model']['sha256'];solves=1
 else:
  try:A.fit(p)
  except AssertionError as e:assert str(e)==r['qualification_failure']
  else:raise AssertionError('Rejected construction unexpectedly qualified')
  solves=0
 rows=G.rows(OUT);rep=G.rows(OUT,'replays.jsonl');by={v['path']:v for v in rows}
 for row in rows:
  a=E.N.S.arrays(row['path']);assert sha(row['path'])==row['sha256'] and int(a['native_counts'].sum())==row['physics'];assert all(digest(v)==row['array_hashes'][k] for k,v in a.items())
  if row['full']:F.M.audit_full_history(p,row,a)
 for row in rep:assert row['exact'] and row['array_hashes']==by[row['reference_path']]['array_hashes'] and row['integration_sha256']==by[row['reference_path']]['integration_sha256']
 if rows:
  aa=E.N.S.arrays(rows[0]['path']);comparison=teacher_physical_match(aa,p);assert comparison==read(OUT/'recovery_physical_comparison.json')
 if r['panel'] and r['panel']['all13_pass']:
  assert len(rows)==len(rep)==13 and comparison['exact'];assert all(v['success'] and not v['collision'] for v in rows)
  for row in rows[1:]:
   previous=next(v for v in p['protected_rows'] if v['job_id']==row['job_id']);assert row['array_hashes']==previous['array_hashes'] and row['integration_sha256']==previous['integration_sha256']
 assert len(rows)==len(rep) and s['physics']==sum(v['physics'] for v in rows+rep)<=p['max_physics'] and s['commands']==(len(rows)+len(rep)+4 if rows else 0)<=p['max_commands']
 save(OUT/'audit.json',dict(state='PASS',pid=os.getpid(),new_native=0,new_Adam=0,independent_closed_form_rebuilds=solves,finite_model_or_rejection_exactly_rebuilt=True,full_recovery_density_hidden_reward_audited=bool(rows),compact12_full120_not_persisted=True,all_native_array_and_integration_replays_exact=True,physics=s['physics'],commands=s['commands'],accepted=False,field24=None));print('T546_AUDIT_PASS',flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
