"""Build a small replay index and test the full recorded-history audit chain."""
from pathlib import Path
import copy,time,json,hashlib
import history_contract as C
ROOT,R,np=C.ROOT,C.R,C.np
DOC=ROOT/'stage/docs/day_20261117/t619'

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def config(path):
 p=read(path);base=read(p['actor_configuration']);base.update(p);return base

def main():
 C.setup_readonly();R.torch.set_num_threads(1);R.torch.use_deterministic_algorithms(True)
 cases=[];p=config(ROOT/'stage/runs/t618_terminal_signed_pulse/protocol.json')
 for row in read(ROOT/'stage/runs/t615_original_start_drift_pair/summary.json')['rows']:
  cases.append((p,row,'student',None,None))
 for row in R.G.rows(ROOT/'stage/runs/t618_terminal_signed_pulse'):
  reference=R.E.N.S.arrays(p['candidate_prefix_source']);cases.append((p,row,'finite_pulse',reference,None))
 q=config(ROOT/'stage/runs/t612_baseline_teacher_recovery/protocol.json')
 row=read(ROOT/'stage/runs/t612_baseline_teacher_recovery/summary.json')['rows'][0]
 cases.append((q,row,'suffix_teacher',R.E.N.S.arrays(q['candidate_prefix_source']),q['handover_teacher']))
 # Original T550 has only its real64 memory, unlike later residual Actors.
 f=ROOT/'stage/runs/t579_frozen_progress_attribution';old=next(r for r in R.G.rows(f) if r['success'] and r['request']['model']==p['base_model']['path'])
 cases.append((p,old,'student',None,None))
 results=[];negatives=[];manifest=[]
 for pc,row,mode,ref,teacher in cases:
  assert sha(row['path'])==row['sha256'];a=R.E.N.S.arrays(row['path']);assert all(C.M.digest(v)==row['array_hashes'][key] for key,v in a.items())
  detail=C.validate(pc,a,row,mode,ref,teacher);detail.update(path=row['path'],sha256=row['sha256'],policy_hash=row['policy_hash']);results.append(detail)
  # Each corruption has a fresh derived view: original bytes are never edited.
  for key in ['actions','ownership','task_goal_xy','history_poses','end_flags']:
   bad=dict(a);bad[key]=a[key].copy()
   if key=='actions':bad[key][0,0]+=.02
   elif key=='ownership':bad[key][0]=1-bad[key][0]
   elif key=='task_goal_xy':bad[key][0]+=.1
   elif key=='history_poses':bad[key][1,0,0]+=.1
   else:bad[key][0,0]=1
   try:C.validate(pc,bad,row,mode,ref,teacher)
   except AssertionError:negatives.append(dict(path=row['path'],field=key,rejected=True))
   else:raise AssertionError(('Corruption not rejected',row['path'],key))
  # No random row shuffle: the index preserves complete episode order and
  # enough data to reconstruct current networks' causal contexts later.
  manifest.append(dict(path=row['path'],sha256=row['sha256'],policy_hash=row['policy_hash'],job_id=row['job_id'],mode=mode,success=row['success'],collision=row['collision'],timeout=row['timeout'],decisions=len(a['states']),physical_steps=row['physics'],native_exact_replay_known=mode=='student' or mode=='suffix_teacher',executed_action_array='actions',behavior_location_array='locations',goal_array='task_goal_xy',held_physical_steps_array='native_counts',terminal_collision_success_timeout='end_flags',mixed_not_student_onpolicy=mode!='student',failed_data_permitted_for_reward_learning=True,failed_data_not_successful_BC=not row['success']))
 assert len(results)==9 and len(negatives)==45 and not R.torch.cuda.is_initialized()
 index=dict(state='IMMUTABLE_REPLAY_INDEX_NOT_TRAINING',episodes=manifest,deduplicate_key='source full-array digest plus policy hash; repeated native replay is not independent new data',native_steps_are_not_decision_steps=True,gamma_native=p['gamma_native'],native_step_seconds=.002,current_policy_hidden_must_be_recomputed=True,reset_hidden_at_arbitrary_fragment_forbidden=True,actor_updates=0,critic_updates=0,new_native=0)
 R.save(DOC/'REPLAY_INDEX.json',index)
 R.save(DOC/'PREPARATION_REPORT.json',dict(state='PASS_RECORDED_EXECUTION_AND_AUDIT_CHAIN',cases=results,negative_tests=negatives,total_cases=len(results),negative_count=len(negatives),pure_student_and_permanent_teacher_and_finite_pulse_covered=True,original_t618_native_replay_still_missing=True,old_errors_not_overwritten=True,no_Runtime_or_Pool_or_optimizer_created=True,new_native=0,new_optimizer=0,epoch=time.time()))
 print('T619_RECORDED_HISTORY_CONTRACT_PASS',len(results),len(negatives))

if __name__=='__main__':main()
