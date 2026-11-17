"""Explicit contracts for immutable Stage replay, independent of launch windows.

Only existing arrays are audited. Never creates a simulator, optimizer or model
checkpoint. Executed actions and Actor proposals remain separate evidence.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t618_terminal_signed_pulse'))
import t618_runtime as M
R,N=M.R,M.Native
np=R.np

def setup_readonly():
 R.NIGHT=M.OUT;R.HERE=M.HERE;R.DOC=M.DOC;R.setup(M.OUT,'compare')

def validate(p,a,row,mode,reference=None,teacher=None):
 assert mode in ['student','suffix_teacher','finite_pulse']
 n=len(a['states']);assert n>0 and a['states'].shape==(n,120)
 assert a['actions'].shape==(n,2) and np.isfinite(a['actions']).all()
 assert not a['stochastic'].any(),'This contract only covers these deterministic evidence records'
 assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
 assert int(a['native_counts'].astype(np.int64).sum())==row['physics']
 flags=a['end_flags'];assert not flags[:-1].any() and bool(flags[-1].any())
 assert list(flags[-1].astype(bool))==[row['collision'],row['success'],row['timeout']]
 assert abs(float(np.linalg.norm(a['next_poses'][-1,:2]-a['task_goal_xy']))-row['remaining_m'])<1e-12
 actor_commands=a['raw_actions'].clip(-1,1).astype(np.float32)
 if mode=='student':
  assert not a['ownership'].any() and not row.get('teacher',False)
  assert np.array_equal(a['actions'],actor_commands)
 else:
  assert reference is not None;k=int(p['teacher_first'])
  for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times','hold_sha256']:
   assert np.array_equal(a[key][:k],reference[key][:k]),key
  for key in ['states','poses','history_poses','history_goals','history_times']:
   assert np.array_equal(a[key][k],reference[key][k]),key
  assert np.array_equal(a['initial_native_integration'],reference['initial_native_integration'])
  if mode=='suffix_teacher':
   assert teacher is not None
   expected=np.arange(n)>=k;assert np.array_equal(a['ownership'],expected.astype(np.uint8))
   assert np.array_equal(a['actions'][:k],actor_commands[:k])
   assert np.array_equal(a['actions'][k:],a['teacher_actions'][k:])
   # The teacher reconstructs its own memory over the actual observations.
   sys.path.insert(0,str(ROOT/'stage/experiments/t617r_terminal_result_recovery'))
   import t617r_native as T
   trace=T.TeacherTrace(dict(p,handover_teacher=teacher))
   for x in a['states']:trace.propose(x)
   assert all(np.array_equal(v,a[key]) for key,v in trace.arrays().items())
  else:
   delta=row['request']['job']['steering_delta']
   expected=(np.arange(n)>=k)&(np.arange(n)<k+p['pulse_decisions'])&(delta!=0)
   assert np.array_equal(a['ownership'],expected.astype(np.uint8))
   commands=a['locations'].copy();commands[expected,1]+=delta
   assert np.array_equal(a['actions'],commands.clip(-1,1).astype(np.float32))
 # Read-only proposal view checks the original network computation. Above we
 # separately check every actual forced command and its genuine ownership.
 view=dict(a,actions=actor_commands,ownership=np.zeros(n,np.uint8))
 N.Previous.inspect(p,[(view,dict(row,teacher=False))])
 return dict(decisions=n,physics=row['physics'],mode=mode,success=row['success'],
             actual_commands_separate_from_proposals=True,full_actual_history=True,
             all_actor_memories_rewards_observations_checked=True)
