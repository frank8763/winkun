from pathlib import Path
import sys,time,os,fcntl,traceback,shutil
import t623_native as N
sys.path.insert(0,str(N.ROOT/'stage/experiments/t620_recurrent_prior_td3'))
import t620_runtime as R
L,torch,np=N.L,N.torch,N.np
ROOT=N.ROOT;HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t623_common_continuation_branches';DOC=ROOT/'stage/docs/day_20261117/t623'
read,save,sha=L.read,L.save,L.sha
def plan():
 p=read(HERE/'protocol.json');c=read(p['actor_configuration']);c.update(p);c.update(out=str(OUT),child_bytes=p['max_bytes']);return c
def verify(p,full=False):
 assert p['authorization_approved'] and sha(HERE/'protocol.json')==sha(OUT/'protocol.json') and time.time()<p['absolute_all_stop']
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized() and set(os.sched_getaffinity(0))<=set(p['stage_cpus'])
 assert sum(f.stat().st_size for f in (ROOT/'stage/runs').rglob('*') if f.is_file())<2**30 and shutil.disk_usage(ROOT).free>=40*2**30
 assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<=p['max_bytes']
 for f,h in (p['inputs'] if full else p['protected_files']).items():assert sha(f)==h,f
def setup():
 R.R.NIGHT=OUT;R.R.HERE=HERE;R.R.DOC=DOC;R.R.setup(OUT,'compare');R.R.Base.verify=verify;R.R.verify=verify;R.R.P.frozen=L.frozen
def metrics(p,rows):
 result=[];k=p['branch_decision']
 for r in rows:
  a=R.R.E.N.S.arrays(r['path']);acc=0.
  for i in range(len(a['rewards'])-1,k-1,-1):acc=float(a['rewards'][i])+p['gamma_native']**int(a['native_counts'][i])*(not bool(a['end_flags'][i].any()))*acc
  result.append(dict(delta=r['request']['job']['steering_delta'],return_from_branch=acc,success=r['success'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],branch_action=a['actions'][k].tolist(),path=r['path']))
 return dict(rows=result,ranking_descending=[z['delta'] for z in sorted(result,key=lambda z:-z['return_from_branch'])],fixed_single_control_then_same_T608=True,not_deployment_success=True)
def execute(rt,p,reference=None):
 B=R.R.Base;jobs=p['jobs'];rt.role='EXACT3' if reference else 'COMMON_CONTINUATION3';rt.guard();assert rt.commands+3+3<=p['max_commands'];assert sum(rt.counts)+sum(j['case']['step_cap'] for j in jobs)<=p['max_physics'];rt.commands+=3;req=[dict(model=p['student_initial']['path'],job=j,seed=None,query=False,full_path=None if reference else str(OUT/'episodes'/f'full_{i}.npz')) for i,j in enumerate(jobs)];values=rt.pool.request(req)
 for i,((a,r),j,request) in enumerate(zip(values,jobs,req)):save(OUT/f'{rt.role}_{i}_raw.json',dict(request=request,native_result=r))
 out=[]
 for i,((a,r),j,request) in enumerate(zip(values,jobs,req)):
  path=Path(reference[i]['path']) if reference else Path(request['full_path']);assert path.exists();r=B.envelope(j,r,a,p['field_limit_m']);r.update(role=rt.role,seed=None,request=request,loaded_model_sha256=p['student_initial']['sha256'],model_sha256=p['student_initial']['sha256'],full=True)
  if reference:
   old=reference[i];assert r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail'];r.update(reference_path=old['path'],exact=True,independent_new_sample=False);rt.replays.append(r);name='replays.jsonl'
  else:r.update(path=str(path),sha256=sha(path));rt.rows.append(r);name='cases.jsonl'
  B.append(name,r);out.append((a,r))
 return out
def main():
 p=plan();setup();verify(p,True);R.R.E.N.offline();R.R.E.L.install_stop_handlers();rt=R.R.Base.Runtime(p);rt.commands=3;rt.counts=N.G.R.M.mp.get_context('spawn').RawArray('q',4);rows=[];error=None
 try:
  rt.pool=N.Pool(dict(p,native_stop_epoch=rt.deadline,phase='student'),rt.counts,rt.guard);vv=execute(rt,p);N.inspect(p,vv);rows=[r for a,r in vv];save(OUT/'METRICS.json',metrics(p,rows));execute(rt,p,rows)
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=3
  save(OUT/'summary.json',dict(state='FAILED' if error else 'COMPLETE',error=error,rows=rows,physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,new_optimizer=0));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert error is None,error
def audit():
 p=plan();setup();verify(p,True);R.R.G.stopped(OUT);s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=R.R.G.rows(OUT);reps=R.R.G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==3 and s['commands']==12
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=R.R.E.N.S.arrays(r['path']);assert all(L.digest(v)==r['array_hashes'][k] for k,v in a.items());N.inspect(p,[(a,r)])
 for r,old in zip(reps,rows):assert r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256']
 actual=metrics(p,rows);assert actual==read(OUT/'METRICS.json') and s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics']
 pred=read(DOC/'PREDICTIONS.json');save(OUT/'COMPARISON.json',dict(predicted_ranking=pred['ranking_descending'],actual_ranking=actual['ranking_descending'],exact_ranking_match=pred['ranking_descending']==actual['ranking_descending'],actual=actual['rows'],single_state_not_global_Q_certification=True,critic_behavior_MC_continuation_mismatch_explicit=True))
 save(OUT/'audit.json',dict(state='PASS',full_actual_history_memory_reward_and_single_action_checked=True,three_exact_array_and_integration_replays=True,new_optimizer=0,new_native=0))
if __name__=='__main__':globals()[sys.argv[1]]()
