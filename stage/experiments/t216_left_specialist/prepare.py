"""Register the missing no-retention counterfactual, preserving all incumbents."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t216_left_specialist'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T216'
    prior=json.loads((ROOT/'stage/runs/t215_curvature_candidate_frozen/summary.json').read_text());assert prior['all_workers_stopped']
    plan=json.loads((ROOT/'stage/experiments/t211_guarded_progress/protocol.json').read_text())
    plan.update(experiment='T216',seed=2026103211,arms=['left_only'],terminal_distance_coefficients={'left_only':.3},
        max_rounds=3,training_episodes_per_arm_per_round=4,max_episodes=54,max_physics=1050000,
        wall_seconds=600,final_reserve_seconds=120,final_bytes=int(9.5*1024**2),peak_bytes=int(10.5*1024**2),
        retention_weight=0.,max_parent_anchor_KL=1e9,max_balanced_anchor_MSE=1e9)
    plan.pop('preflight_rows',None);plan.pop('preflight_contract',None)
    plan['method_contract']=('Singleleftspecialist researchcopy fromT194,full574210trainableMLP andnewindependentV, originalLEFTstarts/rewardterminaldistance.3/nativeGaussian/physicaldiscount/MCbaseline/fullbatchPPO/Adam6proposals/behaviorKL.01/line-searchsameasT211terminal_progress. '
        'ONLYretentionmechanismdisabled:old7MSEweight0,oldparentKL/MSEacceptancegatesdisabled(1e9sentinels),old7closedloopfailuredoesnotstop; old7stillmeasured/fullyreported. '
        'Same seed2026103211 and4episodesperroundasT211matchedfirst3rounds;initialVsame, first4newtrajectoriesmustexactlymatchT211control states/actions/means/logp/counts/reward beforeanyupdate. '
        'Allnewdata freshlycollected,nooldT211trainingdatareuse,oldarraysusedidentitycheckonly. '
        'PerbatchbehaviorKL/surrogateimprovement andfullAdamrollbackstillrequired. '
        'No oldQ/CNN/GRU/newinputs/map/planner/waypoints/runtimeTeacher. T159/T194read-only; nospecialistpromotion. '
        'Thisisretentionmechanismablation,notindependentmulti-seed evidence ornewalgorithmnovelty. 12episodescannotruleoutlearningwithlargerbudgetorproveunreachability.')
    plan['accounting_contract']=('3rounds*4=12freshLEFTtrainingepisodes<=280008trainingphysics; '
        'parent8+perroundold8*3+finalreloadold8=40evaluationepisodes plus1replay=>53commands, max54/1.05Mallphysics/600s,last120sevalreserve. '
        'Max18Adamproposals,acceptedseparate. StoponfirstcenterLEFTsafeGOAL,anyerror/guard or3rounds; '
        'doNOTstoponold7regressionbecauseisolatedspecialist;mustreportit. No local54grid,incumbentchangeorautomaticdistillation/training. '
        'Keep3fullbehaviorexports,allfailuretraces,finalV;nooptimizerexports. Ownfinal9.5MiBpeak10.5MiB/globalnewruns1GiB/free40GiB. '
        'No duplicate/resume/budgetextension;anyLEFTspecialistsuccessrequiresseparateexactreloadandconsolidationprotocol beforewhole-taskclaim.')
    control_rows=[json.loads(s) for s in (ROOT/'stage/runs/t211_guarded_progress/rollouts.jsonl').read_text().splitlines()]
    plan['initial_identity_controls']=[r for r in control_rows if r['arm']=='terminal_progress' and r['round']==0 and r['role']=='training'];assert len(plan['initial_identity_controls'])==4
    plan['references']=['https://arxiv.org/abs/1707.06347','https://arxiv.org/html/1705.10528','https://arxiv.org/abs/1511.06295']
    files=list(HERE.glob('*.py'))+[Path(plan[k]) for k in ('checkpoint','retention_targets')]+[Path(x) for x in plan['task_snapshots']]+[Path(r['trace']) for r in plan['initial_identity_controls']]+[
        ROOT/'stage/experiments/native_update_guard.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',ROOT/'stage/runs/t215_curvature_candidate_frozen/summary.json']
    plan['input_hashes']={str(p):sha(p) for p in files}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','global_runs_MiB':used/1024**2,'protocol_sha256':sha(HERE/'protocol.json')}))
if __name__=='__main__':main()
