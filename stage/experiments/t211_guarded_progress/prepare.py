"""User-authorized successor to failed T210; never restart old experiment."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t211_guarded_progress'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T211'
    prior=json.loads((ROOT/'stage/runs/t210_full_parameter_progress/summary.json').read_text())
    assert prior['all_workers_stopped'] and prior['state']=='FAILED'
    old=json.loads((ROOT/'stage/experiments/t210_full_parameter_progress/protocol.json').read_text())
    plan={k:old[k] for k in ('checkpoint','retention_targets','primary_cases','task_snapshots','arms','terminal_distance_coefficients','actor_lr','value_lr','clip_ratio','grad_clip','retention_weight','robustness_xy','robustness_yaw')}
    plan.update(experiment='T211',seed=2026103211,max_rounds=4,training_episodes_per_arm_per_round=4,workers=4,
                actor_attempts_per_round=6,value_epochs=3,minibatch=256,wall_seconds=900,final_reserve_seconds=180,
                max_episodes=200,max_physics=4700000,final_bytes=30*1024**2,peak_bytes=34*1024**2,pending_bytes=2*1024**2,
                max_behavior_mean_KL=.01,max_parent_anchor_KL=.01,max_balanced_anchor_MSE=.00025,
                minimum_objective_improvement=1e-8,step_factors=[1.,.5,.25,.125,.0625,.03125,.015625],
                batch_mean_tolerance=1e-5,batch_logp_tolerance=1e-4)
    plan['method_contract']=('T194 warm-start full574210parameter MLP, fixednative truncatedGaussian sigma(.35,.30),fresh independentV. '
        'Only inter-arm variable terminaldistance coefficient0/.3 atALL real goal/contact/timeouts asT210. NoCNN/GRU/newobservations/oldQ/curriculum. '
        'Complete fresh originalLEFT rollouts only. CompleteMC-minus-previousV standardizedadvantages. Fullrollout clippedPPO surrogate minus1*balancedold7MSE, '
        'atmost6Adam proposals perround. Acceptbeforeuse viafactors1to1/64: entirebatch regularizedsurrogate mustimprove>=1e-8, '
        'exact nativeTruncGaussianKL tocurrentbehavior mean<=.01, maxanchorKL toT194<=.01,balancedanchorMSE<=.00025. '
        'Rejectall restoresparametersANDAdam; exceptionsrollback. Accepted interpolation retains proposalAdam moments, scales effectivecurrentstep; nominalLRunchanged. '
        'KL anchorbounds finite-stateonly, notsafety/retention/monotonicreturn guarantees or fullTRPO implementation. Fullbatch andacceptancechangedversusT210 engineeringpilot, notcausalguard-onlyablation. '
        'Behavior checkeverybatch: scalarforwardrecordedmean andrecordeddensity errors exactly0; float32batchmean<=1e-5/logp<=1e-4 preregistered. '
        'NewV fittedcompleteMC detached afterActor, noactionQgradient. Allstoredfailedtraces/weights retained, no optimization ofoldT210 data duringonlinelearning.')
    plan['accounting_contract']=('4rounds*4LEFT*2arms=32trainingepisodes<=746688trainingphysics; '
        'parent8, perroundold8*2*4=64, finalreloadold8*2=16,2exacttrainingreplays,selectedrepeatold8+54grid<=200commands/4.7Mphysics. '
        'Atmost48Adamproposals,acceptedupdates separately; line-searchforwardcalls notoptimizersteps. '
        'Stopbotharms afterfirstpairedround anyold7fails orcenterLEFTsafeGOAL, oranyerror/guard; no selectingbesthistoricalcheckpoint. '
        'Finalcurrentserializedpoliciesold8 first, thenrepeat/54gridonlyifall8pass. Missingfinaleval reportedmissing. '
        '900s,last180sreserve,ownfinal30MiBpeak34MiB/globalnewruns1GiB/free40GiB. '
        'Keep every fullbehaviorpolicy and alltrainingobservations/actions/densities/failures; finalV saved,nooptimizerstateexports or exactresumeclaim. '
        'No duplicate/resume/budgetextension/nexttrialautolaunch, T159/T194unchanged/noautopromotion; no5mclaimwithoutgates.')
    plan['references']=['https://proceedings.mlr.press/v37/schulman15.html','https://arxiv.org/abs/1707.06347','https://arxiv.org/abs/2005.12729',
        'https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py','https://arxiv.org/html/2210.04839']
    previous_rows=[json.loads(s) for s in (ROOT/'stage/runs/t210_full_parameter_progress/rollouts.jsonl').read_text().splitlines()]
    plan['preflight_rows']=[r for r in previous_rows if r['role']=='training' and ((r['round']==0 and r['arm']=='original') or (r['round']==1 and r['id'].endswith('train0')))]
    plan['preflight_contract']='Atmost90s/1MiB/0physics; one disposable fresh fullbatchAdam proposal on archivedT210 data only, neverexported/transferred toonlinelearner. VerifybehaviorofparentANDupdated models; mutateddensitynegativecontrol; analyticKL/anchorbounds/objectiveimprovement beforeaccept. Notnavigationtraining orGOAL.'
    files=[Path(plan[k]) for k in ('checkpoint','retention_targets')]+[Path(p) for p in plan['task_snapshots']]+list(HERE.glob('*.py'))+[
        ROOT/'stage/experiments/native_update_guard.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py',
        ROOT/'stage/runs/t210_full_parameter_progress/summary.json',ROOT/'stage/runs/t210_full_parameter_progress/posthoc_update_audit.json']
    files+=list({Path(r[k]) for r in plan['preflight_rows'] for k in ('trace','weights')})
    plan['input_hashes']={str(p):sha(p) for p in files}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','protocol_sha256':sha(HERE/'protocol.json'),'global_runs_MiB':used/1024**2}))
if __name__=='__main__':main()
