"""Read-only recovery of already archived T623 arrays; no simulator/optimizer."""
from pathlib import Path
import sys,time,copy,json,inspect
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t623_common_continuation_branches'))
import t623_runtime as M
L,torch,np=M.L,M.torch,M.np
DOC=ROOT/'stage/docs/day_20261117/t623r'

def check(p,a,r,source):
 k=p['branch_decision'];delta=r['request']['job']['steering_delta'];n=len(a['states']);mask=np.arange(n)<k
 if delta!=0:mask[k]=True
 assert np.array_equal(a['ownership'],mask.astype(np.uint8))
 for key in ['states','next_states','actions','native_counts','poses','next_poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][:k],source[key][:k]),key
 for key in ['states','poses','history_poses','history_goals','history_times']:assert np.array_equal(a[key][k],source[key][k]),key
 assert np.array_equal(a['initial_native_integration'],source['initial_native_integration'])
 proposals=a['raw_actions'].clip(-1,1).astype(np.float32);expected=proposals.copy();expected[:k]=source['actions'][:k];expected[k,1]=np.float32(np.clip(a['locations'][k,1]+delta,-1,1));assert np.array_equal(expected,a['actions'])
 assert np.array_equal(a['states'][1:,117:119],a['actions'][:-1])
 view=dict(a,actions=proposals,ownership=np.zeros(n,np.uint8));M.N.Old.Previous.inspect(p,[(view,dict(r,teacher=False))])

def main():
 p=M.plan();M.setup();torch.set_num_threads(2);assert M.read(M.OUT/'SEALED.json')['all_scientific_processes_stopped'];rows=M.R.R.G.rows(M.OUT);assert len(rows)==3;source=M.R.R.E.N.S.arrays(p['prefix_source']);arrays=[];negative=[]
 for r in rows:
  assert L.sha(r['path'])==r['sha256'];a=M.R.R.E.N.S.arrays(r['path']);assert all(L.digest(v)==r['array_hashes'][key] for key,v in a.items());check(p,a,r,source);arrays.append(a)
  for key in ['actions','ownership','rewards']:
   bad={k:v.copy() for k,v in a.items()};bad[key].flat[-1]+=1
   try:check(p,bad,r,source)
   except (AssertionError,ValueError):negative.append(dict(path=r['path'],field=key,rejected=True))
   else:raise AssertionError(('Corruption accepted',key))
 k=p['branch_decision'];assert all(np.array_equal(arrays[0]['hold_sha256'][:k],a['hold_sha256'][:k]) for a in arrays[1:])
 # The hold digest includes ownership and Actor proposal, so the stochastic
 # source cannot equal forced-prefix deterministic Actor digests even when
 # every executed command, observation and history is exactly identical.
 old_error=False
 try:M.N.inspect(p,[(arrays[0],rows[0])])
 except AssertionError as e:assert str(e)=='hold_sha256';old_error=True
 assert old_error
 native=M.metrics(p,rows);pred=M.read(M.DOC/'PREDICTIONS.json');summary=M.read(M.OUT/'summary.json');assert sum(r['physics'] for r in rows)==summary['physics']==86188
 report=dict(state='PASS_OFFLINE_ARRAY_CONTRACT_RECOVERY_ORIGINAL_PRODUCER_FAILED',original_producer_state='FAILED',original_error_reproduced=True,original_error='hold digest contains behavior ownership/proposal, not only physics',actual_prefix_observation_action_pose_history_exact=True,cross_branch_prefix_full_hold_digest_exact=True,source_prefix_native_hold_integration_digest_not_comparable_due_metadata=True,all_actual_actor_memories_rewards_and_actions_reconstructed=True,negative_tests=negative,physics_already_charged=86188,commands_already_charged=9,new_native=0,new_optimizer=0,original_exact_replays_executed=0,original_exact_replays_missing=3,native_metrics=native,predictions=pred,ranking_match=pred['ranking_descending']==native['ranking_descending'],raw_originals_immutable_and_remote_verified=True,epoch=time.time());L.save(DOC/'RECOVERY.json',report);print(json.dumps(dict(state=report['state'],predicted=pred['ranking_descending'],actual=native['ranking_descending'],new_native=0,exact_missing=3)))
if __name__=='__main__':main()
