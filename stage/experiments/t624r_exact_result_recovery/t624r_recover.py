"""Unique-module, zero-native recovery after all three exact returns were saved."""
from pathlib import Path
import sys,importlib.util,time,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'stage/experiments/t623_common_continuation_branches'))
import t623_runtime as M
spec=importlib.util.spec_from_file_location('t623r_readonly_contract_unique',ROOT/'stage/experiments/t623r_branch_contract_recovery/recover.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
L=M.L;OUT=ROOT/'stage/runs/t624_exact_branch_completion';DOC=ROOT/'stage/docs/day_20261117/t624r'
def main():
 assert L.read(OUT/'SEALED.json')['all_scientific_processes_stopped'];p=M.plan();M.setup();s=L.read(OUT/'summary.json');refs=L.read(ROOT/'stage/docs/day_20261117/t624/REFERENCE_ROWS.json');reps=M.R.R.G.rows(OUT,'replays.jsonl');assert len(reps)==3 and s['commands']==9 and s['physics']==sum(z['physics'] for z in reps)==86188
 source=M.R.R.E.N.S.arrays(p['prefix_source'])
 for old,r in zip(refs,reps):
  assert L.sha(old['path'])==old['sha256'] and r['exact'] and r['array_hashes']==old['array_hashes'] and r['integration_sha256']==old['integration_sha256'];a=M.R.R.E.N.S.arrays(old['path']);assert all(L.digest(v)==old['array_hashes'][key] for key,v in a.items());C.check(p,a,old,source)
 aff=L.read(OUT/'ACTUAL_AFFINITY.json');assert aff['actual']==[[8],[10],[12]]
 native=M.metrics(p,refs);pred=L.read(M.DOC/'PREDICTIONS.json');report=dict(state='PASS_INDEPENDENT_EXACT_RESULT_RECOVERY_ORIGINAL_PRODUCERS_FAILED',original_T623='FAILED_hold_digest_semantic_assertion',original_T624='FAILED_generic_recover_module_collision_after_three_saved_exact_returns',new_native=0,new_optimizer=0,exact_replays_completed=3,all_arrays_and_integrations_and_field_detail_exact=True,all_actual_actor_memories_rewards_ownership_actions_reconstructed=True,actual_CPU_worker_affinity=aff['actual'],T623_actual_physics=86188,T624_actual_physics=86188,total_original_plus_exact_physics=172376,total_commands=18,native_metrics=native,predicted_ranking=pred['ranking_descending'],actual_ranking=native['ranking_descending'],ranking_match=pred['ranking_descending']==native['ranking_descending'],retained_model_unchanged=L.sha(p['student_initial']['path'])==p['student_initial']['sha256'],no_global_navigation_gain=True,epoch=time.time());assert report['retained_model_unchanged'];L.save(DOC/'RECOVERY_AUDIT.json',report);print(json.dumps(dict(state=report['state'],exact_replays=3,predicted=pred['ranking_descending'],actual=native['ranking_descending'],new_native=0)))
if __name__=='__main__':main()
