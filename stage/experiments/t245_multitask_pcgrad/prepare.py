from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t245_multitask_pcgrad'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicateT245'
    s=json.loads((ROOT/'stage/runs/t243_induced_retention_repair/summary.json').read_text());assert s['all_workers_stopped'] and s['development_safeGOAL']==209
    old=json.loads((ROOT/'stage/experiments/t239_ar_policy_gradient/protocol.json').read_text());best=json.loads((ROOT/'stage/runs/t239_ar_policy_gradient/summary.json').read_text());rec=json.loads((ROOT/'stage/runs/t242_induced_state_recovery/summary.json').read_text());assert best['development_safeGOAL']==210 and rec['state']=='COMPLETE'
    plan=dict(old);plan.update(experiment='T245',checkpoint=best['export']['path'],seed=2026103301,induced_states=rec['dataset']['path'],hard_jobs=[j for j in old['development_jobs'] if j['id'] in ['dev_task6_00','dev_task6_12','dev_task6_21','dev_task6_24']],max_anchor_q99=old['max_anchor_error'],max_commands=464,max_physics=10826976,wall_seconds=1800,final_bytes=10*1024**2,peak_bytes=10*1024**2,
      contract='NewactualRLhypothesis afterT243auxiliaryMSEfitfailed209/211; no furtherBC/weightgrid. StartretainedbestT239210/211, fourcriticaldevelopmentresets00/12/21/24 each4freshseeds per4rounds=64learnerrollouts+64independent exact regeneration; no frozen exploration control/no statistical superiority claim. Same originalnative reward/ARsigma,rho/historyconditionallikelihood/PPOsurrogate/lr/4Adamattempts/guardrollback. FullActor taskwiseMCgradients (same-resetother3LOObaseline), fixedseedPCGradnegativeprojection among4taskreturngradients, THENadd fixedT239nativefunctionretentiongradient. Denseactual210safeT239inducedstates replaceoldT228anchors; q99same redundant.02 asT240, MSE4e-6/max.02/k3.02/maxlogratio1unchanged. Mixedhypothesis(curriculum/coverage/gradient surgery), NOTsinglefactorablation/PCGradimprovementproof/newalgorithmnovelty. Synthetic projectionchecks beforelaunch; everyliveupdate independentlyshadowrecomputedinclrandomorder/Adam; initial4frozencasesexactT239match. Eachroundold8centers+4criticalcases; anyoldcenterloss stopsfurthertraining. LastbudgetActor notbestintermediate; ALL211validdevelopmentthenonlyifallsafe64unseenheldout. Up to455plannedcommands/max464,10.826976Mallphys/1800s/reserve650final/10MiBown/global1GiB/free40GiB. Same singleendtoend113inputnative2output, no map/waypoint/planner/runtimeTeacher/routing/Phase. Oldmodels/evidenceunchanged, noautomaticpromotion/longdistance, noautomaticextraRLrounds/seed/hyperparametergrid.',
      references=old['references']+['https://arxiv.org/abs/2001.06782'])
    assert len(plan['hard_jobs'])==4
    paths=list(HERE.glob('*.py'))+[Path(plan['checkpoint']),Path(plan['induced_states']),ROOT/'stage/runs/t239_ar_policy_gradient/development.jsonl',ROOT/'stage/runs/t239_ar_policy_gradient/summary.json',ROOT/'stage/runs/t243_induced_retention_repair/summary.json'];plan['input_hashes']={**old['input_hashes'],**{str(p):sha(p) for p in paths}}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file());assert used+plan['peak_bytes']<plan['global_bytes'] and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in [HERE/'protocol.json',OUT/'protocol.json']:p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(state='REGISTERED',cases=4,runs_MiB=used/1024**2)))
if __name__=='__main__':main()
