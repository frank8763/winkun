"""Read-only scientific reconstruction of T618; no native/optimizer execution."""
from pathlib import Path
import sys,time,copy
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t618_terminal_signed_pulse'))
import t618_runtime as M
N,R=M.Native,M.R
np=R.np

def inspect(p,a,r):
 k=p['teacher_first'];delta=r['request']['job']['steering_delta'];source=R.E.N.S.arrays(p['candidate_prefix_source']);n=len(a['states']);active=(np.arange(n)>=k)&(np.arange(n)<k+p['pulse_decisions'])&(delta!=0)
 assert np.array_equal(a['ownership'],active.astype(np.uint8)) and not a['stochastic'].any()
 for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hold_sha256']:assert np.array_equal(a[key][:k],source[key][:k]),key
 for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],source[key][k]),key
 assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
 expected=a['locations'].copy();expected[active,1]+=delta;assert np.array_equal(expected.clip(-1,1).astype(np.float32),a['actions'])
 assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
 # Audit a derived view of the Actor's proposals, while the checks above retain
 # the original executed action/mask evidence. Rewards, observations, poses,
 # tapes and all three memories are still the actual recorded arrays.
 proposals=a['raw_actions'].clip(-1,1).astype(np.float32)
 N.Previous.inspect(p,[(dict(a,actions=proposals,ownership=np.zeros(n,np.uint8)),dict(r,teacher=False))])

def main():
 # Initialize only read-only history decoder/model audit dependencies. Do not
 # call the ended experiment's launch/time admission or create a Runtime/Pool.
 p=M.plan();R.NIGHT=M.OUT;R.HERE=M.HERE;R.DOC=M.DOC;R.setup(M.OUT,'compare')
 for path,expected in p['inputs'].items():assert M.sha(path)==expected
 rows=R.G.rows(M.OUT);s=M.read(M.OUT/'summary.json');assert s['state']=='FAILED' and len(rows)==3
 source=R.E.N.S.arrays(p['candidate_prefix_source']);results=[];original_bug=0;negative=0
 for r in rows:
  assert M.sha(r['path'])==r['sha256'];a=R.E.N.S.arrays(r['path']);assert all(M.digest(v)==r['array_hashes'][k] for k,v in a.items())
  try:N.inspect(p,[(a,r)])
  except AssertionError:original_bug+=1
  else:raise AssertionError('Original suffix-only ownership check must reject this pulse schema')
  inspect(p,a,r)
  for field in ['actions','ownership']:
   bad=dict(a);bad[field]=a[field].copy();k=p['teacher_first']
   if field=='actions':bad[field][k,0]+=.01
   else:bad[field][k]=1-bad[field][k]
   try:inspect(p,bad,r)
   except AssertionError:negative+=1
   else:raise AssertionError('Corrupted actual correction evidence accepted')
  if r['signed_steering_delta']==0:assert all(np.array_equal(a[k],v) for k,v in source.items())
  goal=a['task_goal_xy'];positions=np.vstack([a['poses'][:,:2],a['next_poses'][-1,:2]]);distance=np.linalg.norm(positions-goal,axis=1);j=int(distance.argmin());times=np.r_[0,np.cumsum(a['native_counts'].astype(np.int64))]*.002
  flags=a['end_flags'][-1].astype(bool);assert list(flags)==[r['collision'],r['success'],r['timeout']]
  assert abs(distance[-1]-r['remaining_m'])<1e-12 and int(a['native_counts'].astype(np.int64).sum())==r['physics']
  results.append(dict(delta=r['signed_steering_delta'],success=r['success'],collision=r['collision'],timeout=r['timeout'],remaining_m=r['remaining_m'],minimum_recorded_goal_m=float(distance[j]),minimum_recorded_physical_time_s=float(times[j]),physics=r['physics'],path=r['path'],sha256=r['sha256'],full_actual_history_reward_all_memories_and_pulse_formula_checked=True,native_exact_replay=False,qualified_training_data=False))
 workers=M.read(M.OUT/'worker_generations.json');pids=[z['parent'] for z in workers]+[i for z in workers for i in z['workers']];assert all(not Path('/proc',str(i)).exists() for i in pids)
 assert sum(r['physics'] for r in rows)==s['physics']==192336 and s['commands']==7
 out=dict(state='OFFLINE_RECOVERY_PASS_ORIGINAL_PRODUCER_FAILED',original_bug_reproduced_count=original_bug,negative_corruptions_rejected=negative,results=results,all_original_full_arrays_SHA=True,zero_branch_all_historical_arrays_exact=True,no_native_replay_performed=True,no_successful_recovery_label=True,original_failure_and_source_preserved=True,new_native=0,new_optimizer=0,original_physics=s['physics'],original_commands=s['commands'],all_scientific_pids_gone=True,epoch=time.time());M.save(ROOT/'stage/docs/day_20261117/t618r/RECOVERY_AND_REGRESSION.json',out);print(out)

if __name__=='__main__':main()
