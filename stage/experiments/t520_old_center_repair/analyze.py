"""Zero-native/optimizer review of T519 and next-round capacity admission draft."""
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t519_fresh_full_actor_rl'))
import t519_runtime as R
p=R.read(R.OUT/'protocol.json');s=R.read(R.OUT/'pilot/summary.json');seal=R.read(R.OUT/'SEALED.json')
assert all(not Path('/proc',str(i)).exists() for i in seal['scientific_pids']+[R.read(R.OUT/'launch.json')['pid']])
assert R.read(R.OUT/'pilot/audit.json')['state']=='PASS'
assert R.read(ROOT/'stage/docs/long_distance_20261101/release_receipt_T519.json')['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
result=s['result'];model=result['model'];assert R.sha(model['path'])==model['sha256']
q=R.L.A.frozen(p,model['path']);rows=R.G.rows(R.OUT/'pilot');old=[r for r in rows if r['role']=='AUTONOMOUS_FULL32' and not r['job_id'].startswith('t502_')]
assert len(old)==8;failed=[r for r in old if not r['success']];assert len(failed)==1
r=failed[0];assert r['job_id']=='dev_task6_13' and r['timeout'] and not r['collision']
parentrow=next(r0 for r0 in p['keep_rows'] if r0['job_id']==r['job_id']);a=R.E.N.S.arrays(parentrow['path'])
with R.torch.no_grad():
 loc=q.sequence(R.torch.tensor(a['states'])).numpy();diff=loc-a['locations'];i=int(R.np.abs(diff).max(1).argmax())
review=dict(epoch=time.time(),native=0,Adam=0,source_summary_sha256=R.sha(R.OUT/'pilot/summary.json'),source_audit_sha256=R.sha(R.OUT/'pilot/audit.json'),
 candidate=model,candidate_accepted=False,field24=result['field24'],old8=result['old8'],gained_ids=result['gained_ids'],lost_field_ids=result['lost_ids'],
 old_failed_task=r['job_id'],old_failure_type='TIMEOUT_NO_COLLISION',old_final_distance_m=r['remaining_m'],old_final_seconds=r['physics']*.002,
 continuous_or_decision_minimum_distance_UNKNOWN=True,evaluation_full120_NOT_saved=True,
 parent_history_output_diagnostic=dict(location_RMSE_per_axis=R.np.sqrt(R.np.mean(diff**2,0)).tolist(),max_abs_per_axis=R.np.abs(diff).max(0).tolist(),worst_parent_history_time_s=float(a['native_counts'][:i].sum()*.002),not_actual_candidate_closedloop_causal_decomposition=True),
 locked_best=p['parent_model'],locked_best_SHA_unchanged=R.sha(p['parent_model']['path'])==p['parent_model']['sha256'],
 next_draft_only=dict(out_directory_NOT_CREATED=True,new_model_NOT_promoted=True,initial_candidate=model,
 training_task='dev_task6_13',fresh_episodes=4,exact_replays=4,max_PG_steps=16,
 behavior_acceptance='old8all8safeGOAL plus all12T519fieldsuccessIDs retained,field>=12/contact<=9/0invalidcensor; full32+32nativeexact, no actionRMSEhardgate.',
 own_bytes_proposed=6*2**20,actual_global_bytes=R.M.used(),global_reserved=R.M.used()+6*2**20,global_limit=2**30))
assert review['next_draft_only']['global_reserved']<2**30
R.save(ROOT/'stage/docs/day_20261113/T519_REVIEW_AND_NEXT.json',review)
print(json.dumps({k:review[k] for k in ['field24','old8','gained_ids','lost_field_ids','old_failed_task','old_failure_type','old_final_distance_m','parent_history_output_diagnostic','next_draft_only']},ensure_ascii=False))
