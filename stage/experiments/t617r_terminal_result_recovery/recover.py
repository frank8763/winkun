"""Offline recovery of preserved T617 arrays; no native execution or optimization."""
from pathlib import Path
import sys,time,unittest.mock as mock
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t617_terminal_teacher_pair'));import t617_runtime as M
import t617r_native as N

def main():
 p=M.plan();M.setup();rows=[];reg=[]
 for i,job in enumerate(p['teacher_jobs']):
  a=M.R.E.N.S.arrays(M.OUT/f'episodes/full_{i}.npz');raw=dict(policy_hash=p['candidate_model']['policy_hash']);req=dict(model=p['candidate_model']['path'],job=job)
  old=object.__new__(M.Native.Pool);old.p=p
  with mock.patch.object(M.Native.Previous.Pool,'receive',return_value=(a,dict(raw))):
   try:M.Native.Pool.receive(old,i)
   except KeyError as e:assert e.args==('request',)
   else:raise AssertionError('Original seam must fail before envelope attaches request')
  new=object.__new__(N.Pool);new.p=p;new.pending_requests=[dict(model=p['candidate_model']['path'],job=j) for j in p['teacher_jobs']]
  with mock.patch.object(N.Previous.Pool,'receive',return_value=(a,dict(raw))):_,r=N.Pool.receive(new,i)
  assert r['teacher_policy_hash']==job['handover_teacher']['policy_hash'] and 'request' not in r
  r.update(request=req,seed=None,array_hashes={k:M.digest(v) for k,v in a.items()},array_digest=M.digest(a),integration_sha256=M.digest(a['final_native_integration']),path=str(M.OUT/f'episodes/full_{i}.npz'));r['sha256']=M.sha(r['path']);N.inspect(p,[(a,r)])
  flags=a['end_flags'][-1].astype(bool);goal=a['task_goal_xy'];ds=M.np.linalg.norm(a['poses'][:,:2]-goal,axis=1);remaining=float(M.np.linalg.norm(a['next_poses'][-1,:2]-goal));rows.append(dict(teacher=job['handover_teacher'],path=r['path'],sha256=r['sha256'],physics=int(a['native_counts'].astype(M.np.int64).sum()),terminal_flags_collision_success_timeout=flags.tolist(),remaining_m=remaining,minimum_recorded_goal_distance_m=float(min(ds.min(),remaining)),full_candidate_and_teacher_memory_and_rewards_reconstructed=True,original_prefix_exact=True,native_exact_replay=False,training_label_eligible=False));reg.append(dict(teacher_index=i,old_KeyError_request_reproduced=True,new_result_annotation_uses_pending_request=True,teacher_association_matches_issued_request=True))
 workers=M.read(M.OUT/'worker_generations.json');pids=[z['parent'] for z in workers]+[i for z in workers for i in z['workers']]
 out=dict(state='OFFLINE_RAW_RECOVERY_COMPLETE_NO_NATIVE_REPLAY',original_producer_state='FAILED',original_failure_preserved=True,new_native=0,new_optimizer=0,actual_original_physics=sum(r['physics'] for r in rows),actual_original_commands=6,rows=rows,regression=reg,native_exact_replay_missing=True,no_successful_training_labels=True,all_worker_pids_gone=all(not Path('/proc',str(i)).exists() for i in pids),epoch=time.time());assert out['actual_original_physics']==128224 and out['all_worker_pids_gone'];M.save(ROOT/'stage/docs/day_20261117/t617r/RECOVERY_AND_REGRESSION.json',out);print(out)
if __name__=='__main__':main()
