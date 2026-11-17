"""Prepared zero-learning parent/candidate native trajectory pair. Not auto-launched."""
from pathlib import Path
import sys,os,time,fcntl,traceback
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t617_terminal_teacher_pair';DOC=ROOT/'stage/docs/day_20261117/t617'
sys.path.insert(0,str(ROOT/'stage/experiments/t608_current_policy_memory'));import t608_runtime as R
import t617_native as Native
read,save,sha,digest,torch,np=R.read,R.save,R.sha,R.digest,R.torch,R.np

def setup():
 R.NIGHT=OUT;R.HERE=HERE;R.DOC=DOC;R.setup(OUT,'compare');R.verify(plan(),True)
def plan():
 p=read(HERE/'protocol.json');assert p['authorization_approved'];cfg=read(p['actor_configuration']);cfg.update(p);cfg.update(out=str(OUT),child_bytes=p['max_bytes']);return cfg

def metrics(p,rows):
 assert len(rows)==2;out=[]
 for r in rows:
  a=R.E.N.S.arrays(r['path']);k=p['teacher_first'];out.append(dict(teacher=r['request']['job']['handover_teacher'],fixed_handover_decision=k,handover_native_steps=int(a['native_counts'][:k].astype(np.int64).sum()),teacher_recovery_success=R.safe(r),success=r['success'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],physics=r['physics'],mixed_teacher_diagnostic_not_deployment_success=True))
 return dict(rows=out,new_Adam=0,new_RL=0,same_actual_prefix_between_teachers=True)

def main():
 assert OUT.exists() and not (OUT/'launch.json').exists();p=plan();setup();R.E.N.offline();R.E.L.install_stop_handlers();rt=R.Base.Runtime(p);save(OUT/'launch.json',dict(pid=os.getpid(),epoch=time.time(),new_Adam=0,new_RL=0,scope='two fixed teachers after same actual74s candidate prefix plus2exact'));rows=[];state='FAILED';error=None
 try:
  rt.pool=Native.Pool(dict(p,phase='student',native_stop_epoch=rt.deadline),rt.counts,rt.guard)
  vv=rt.request(p['teacher_jobs'],p['candidate_model'],'TERMINAL_TEACHER_PAIR',full=True);Native.inspect(p,vv);rows=[r for a,r in vv];save(OUT/'PROGRESS.json',dict(rows=rows));rt.request(p['teacher_jobs'],p['candidate_model'],'EXACT_TERMINAL_TEACHER_PAIR',reference=rows)
  detail=metrics(p,rows);state='COMPLETE';save(OUT/'METRICS.json',detail)
 except BaseException:error=traceback.format_exc();save(OUT/'failure.json',dict(error=error,raw_preserved=True,no_restart=True))
 finally:
  if rt.pool:rt.pool.close();rt.commands+=2
  save(OUT/'summary.json',dict(state=state,pid=os.getpid(),physics=sum(rt.counts),commands=rt.commands,elapsed_s=time.time()-rt.start,error=error,rows=rows,new_Adam=0,new_RL=0));fcntl.flock(rt.lock,fcntl.LOCK_UN);rt.lock.close()
 assert state=='COMPLETE',error

def audit():
 p=plan();setup();R.G.stopped(OUT);assert not (OUT/'audit.json').exists();s=read(OUT/'summary.json');assert s['state']=='COMPLETE';rows=R.G.rows(OUT);reps=R.G.rows(OUT,'replays.jsonl');assert len(rows)==len(reps)==2 and len(rows)+len(reps)+4==s['commands']==8;by={x['path']:x for x in rows}
 for r in rows:
  assert sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path']);assert all(digest(v)==r['array_hashes'][k] for k,v in a.items());Native.inspect(p,[(a,r)])
 for r in reps:assert r['exact'] and r['array_hashes']==by[r['reference_path']]['array_hashes'] and r['integration_sha256']==by[r['reference_path']]['integration_sha256']
 assert s['physics']==sum(r['physics'] for r in rows+reps)<=p['max_physics'] and metrics(p,rows)==read(OUT/'METRICS.json')
 a,b=[R.E.N.S.arrays(r['path']) for r in rows];k=p['teacher_first']
 for key in ['states','actions','native_counts','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][:k],b[key][:k]),key
 assert np.array_equal(a['states'][k],b['states'][k])
 save(OUT/'audit.json',dict(state='PASS',new_native=0,new_Adam=0,all_full_arrays_and_native_integrations_exact=True,physics=s['physics'],commands=s['commands'],actual_prefix_and_two_teacher_memories_reconstructed=True,not_two_order_decomposition=True))
if __name__=='__main__':globals()[sys.argv[1]]()
