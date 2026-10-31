"""Recover exact T239-induced states, not more independent test samples."""
from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t242_induced_state_recovery'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicateT242'
    a=ROOT/'stage/runs/t239_ar_policy_gradient';b=ROOT/'stage/runs/t240_retention_ablation';sa=json.loads((a/'summary.json').read_text());sb=json.loads((b/'summary.json').read_text());assert sa['all_workers_stopped'] and sb['all_workers_stopped'] and sa['development_safeGOAL']==210 and sb['development_safeGOAL']==209
    rows=list(map(json.loads,(a/'development.jsonl').read_text().splitlines()));extra=next(r for r in map(json.loads,(b/'development.jsonl').read_text().splitlines()) if r['job']['source_reset_id']=='dev_task6_24');assert extra['success'] and not extra['collision']
    jobs=[dict(r['job'],id='recover239_'+r['job']['source_reset_id'],arm='T239',expected=r) for r in rows]+[dict(extra['job'],id='recover240_dev_task6_24',arm='T240',expected=extra)]
    plan=dict(experiment='T242',checkpoints=dict(T239=sa['export']['path'],T240=sb['export']['path']),jobs=jobs,max_commands=212,max_physics=4946808,wall_seconds=900,final_bytes=10*1024**2,peak_bytes=10*1024**2,max_dataset_bytes=8*1024**2,global_bytes=1024**3,
      contract='Frozen exact state reconstruction after T239210/211 and single-factorT240209/211 (fixed24butfailed12/21). ExistingretentionanchorsareolderT228trajectory states, not actualT239210safeinduced trajectories; recovertheseinputstates before anynewhypothesis ofsuccessretention. Exactlyall211T239fixedvaliddevelopmentcases includingfailed24 plusoneT240successful24, samefrozenexport, onefreshgenerationeach must all10arrays+finalintegration exactlymatch original recorded independentprocess. 212commands/4.946808Mallphys/900s/10MiBown/global1GiB/free40GiB. Theseare counted physicalreconstructions, NOT212independent new tests/successes. Keepcomplete statesall211 includingfailure inonecompressed dataset offsets/ids/successlabels; retainoriginaloldcompactaction/pose/rewardtraces as referenced, no duplicateallarrays. T240success full10arrays compactlossless nextstate. Nooptimizer/modelselection/teachercontroller/planner/map/waypoint/Phase/heldoutlabels. Sourcepolicies/failedresultsunchanged. Noautomatictraining/promotion/longdistance.',
      references=['https://proceedings.mlr.press/v15/ross11a.html','https://proceedings.mlr.press/v78/laskey17a.html','https://arxiv.org/abs/1806.05635'])
    paths=list(HERE.glob('*.py'))+[Path(p) for p in plan['checkpoints'].values()]+[a/'summary.json',a/'development.jsonl',b/'summary.json',b/'development.jsonl']+[Path(j['exact_snapshot']) for j in jobs]+[ROOT/'stage/experiments/t229_valid_start_acceptance/valid_support.py',ROOT/'stage/experiments/t223_coverage_repair/repair_support.py',ROOT/'stage/experiments/t219_rnd_runtime/support.py',ROOT/'stage/experiments/t219_rnd_runtime/native_compat/g5b.py'];plan['input_hashes']={str(p):sha(p) for p in paths}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',cases=212,runs_MiB=used/1024**2)))
if __name__=='__main__':main()
