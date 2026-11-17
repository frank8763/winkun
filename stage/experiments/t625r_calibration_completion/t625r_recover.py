"""Finish the original independent data audit without inventing a missing PID."""
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t625_target_continuation_calibration'))
import t625_collect as C
L=C.L;DOC=ROOT/'stage/docs/day_20261117/t625r'
def main():
 p=C.plan();C.setup(p);C.verify(p);root=C.OUT;folder=root/'collect';sealed=L.read(root/'SEALED.json');assert sealed['all_scientific_processes_stopped'];assert all(not Path('/proc',str(i)).exists() for i in sealed['scientific_pids'])
 workers=L.read(folder/'worker_generations.json');ids=[i for g in workers for i in [g['parent'],*g['workers']]];assert all(not Path('/proc',str(i)).exists() for i in ids)
 try:C.M.R.R.G.stopped(folder)
 except KeyError as e:assert e.args==('pid',)
 else:raise AssertionError('Original error no longer reproduced')
 s=L.read(folder/'summary.json');rows=C.M.R.R.G.rows(folder);reps=C.M.R.R.G.rows(folder,'replays.jsonl');assert s['state']=='COMPLETE' and len(rows)==len(reps)==3 and s['commands']==12 and s['physics']==sum(r['physics'] for r in rows+reps)==384672
 for old,r in zip(rows,reps):
  assert L.sha(old['path'])==old['sha256'] and r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'] and r['field_detail']==old['field_detail'];a=C.M.R.R.E.N.S.arrays(old['path']);assert all(L.digest(v)==old['array_hashes'][key] for key,v in a.items());C.New.inspect(p,[(a,old)])
 report=dict(state='PASS',scope='Recovered original independent collection audit; producer FAILED retained',original_missing_pid_reproduced=True,actual_original_pids_verified_stopped=ids,three_original_three_exact=True,full_history_all_Actor_memory_reward_and_action_contract=True,all_saved_arrays_SHA_and_exact_native_integrations=True,physics_already_charged=384672,commands_already_charged=12,new_native=0,new_optimizer=0,epoch=time.time());L.save(DOC/'COLLECTION_RECOVERY_AUDIT.json',report);print('T625R_ORIGINAL_COLLECTION_INDEPENDENT_AUDIT_PASS')
if __name__=='__main__':main()
