"""Index real historical transitions; never fabricate rewards from BC memories."""
from pathlib import Path
import hashlib,json,time
import history_contract as C
ROOT,R,np=C.ROOT,C.R,C.np
DOC=ROOT/'stage/docs/day_20261117/t619'

def read(f):return json.loads(Path(f).read_text())
def sha(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()
def audit_for(path):
 for parent in Path(path).parents:
  if parent==ROOT/'stage/runs':break
  f=parent/'audit.json'
  if f.exists() and read(f).get('state')=='PASS':return dict(path=str(f),sha256=sha(f))
 return None

def main():
 C.setup_readonly();p=C.M.plan();sources=list(p['keep_rows'])+list(p['recovery_rows'])
 sources+=read(ROOT/'stage/runs/t612_baseline_teacher_recovery/summary.json')['rows']
 for child in ['rl_00','rl_01']:
  sources+=read(ROOT/'stage/runs/t608_current_policy_memory'/child/'summary.json')['result']['training_rows']
 sources+=read(ROOT/'stage/runs/t615_original_start_drift_pair/summary.json')['rows']
 index=[];seen=set();duplicates=[]
 for row in sources:
  f=Path(row['path']);assert sha(f)==row['sha256']
  if 'policy_hash' not in row or ('physics' not in row and 'physics_steps' not in row):
   original=next((r for r in R.G.rows(f.parent.parent) if r.get('path')==str(f)),None)
   assert original is not None,('Missing authoritative native row',str(f))
   assert original['sha256']==row['sha256'] and original['job_id']==row['job_id']
   row=original
  a=R.E.N.S.arrays(f)
  required=['states','next_states','actions','native_counts','rewards','poses','next_poses','end_flags','history_poses','history_goals','history_times','initial_native_integration','final_native_integration','task_goal_xy']
  assert all(key in a for key in required);n=len(a['states']);assert a['states'].shape==a['next_states'].shape==(n,120)
  assert a['actions'].shape==(n,2) and len(a['rewards'])==n and a['end_flags'].shape==(n,3)
  assert np.isfinite(a['states']).all() and np.isfinite(a['rewards']).all();assert np.array_equal(a['next_states'][:-1],a['states'][1:])
  assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1]);physics=int(a['native_counts'].astype(np.int64).sum())
  assert physics==row.get('physics',row.get('physics_steps'))
  key=C.M.digest({k:a[k] for k in required})
  if key in seen:duplicates.append(dict(path=str(f),same_scientific_trajectory_key=key));continue
  seen.add(key);audit=audit_for(f)
  index.append(dict(path=str(f),sha256=row['sha256'],scientific_trajectory_key=key,job_id=row['job_id'],behavior_policy_hash=row['policy_hash'],success=row['success'],collision=row['collision'],timeout=row['timeout'],decisions=n,physics=physics,data_partition='historical_prior_NEVER_new_online',has_all_real_RL_fields=True,existing_source_audit=audit,training_eligibility='PRIOR_SCHEMA_AND_EXISTING_AUDIT_VERIFIED' if audit else 'PENDING_SOURCE_AUDIT_NOT_TRAINING_READY',mixed=bool(a['ownership'].any()),behavior_logp_not_used_as_current_policy_logp=True))
 memory=read(p['memory_manifest']);bad=[]
 for row in memory['memories']:
  a=R.E.N.S.arrays(row['path']);missing=[k for k in ['next_states','rewards','end_flags'] if k not in a]
  assert missing
  bad.append(dict(path=row['path'],sha256=row['sha256'],use='BC_ONLY_NOT_RL_TRANSITIONS',missing_RL_fields=missing,native_reference=row['native_reference']))
 report=dict(state='PRIOR_REPLAY_SCHEMA_INDEX_COMPLETE_NOT_TRAINING',full_prior_episodes=index,scientific_duplicates=duplicates,BC_only_memories=bad,new_online_buffer_empty=True,new_online_requires_fresh_new_learner_collection=True,new_native=0,new_optimizer=0,prior_episodes=len(index),BC_only_count=len(bad),total_prior_decisions=sum(z['decisions'] for z in index),no_cached_other_policy_hidden_used_as_current_hidden=True,no_reward_or_next_state_fabricated=True,not_all_source_audits_repeated=True,epoch=time.time())
 R.save(DOC/'RL_PRIOR_AVAILABILITY.json',report);print('PRIOR_SCHEMA_INDEX',len(index),'episodes',report['total_prior_decisions'],'decisions;',len(bad),'BC-only memories;',sum(z['existing_source_audit'] is None for z in index),'pending source audits')

if __name__=='__main__':main()
