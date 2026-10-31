"""Bounded paired navigation pilot with new losslessly reconstructible records."""
from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;OUT=ROOT/'stage/runs/t219_rnd_runtime'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not OUT.exists(),'No duplicate T219'
    rnd=json.loads((ROOT/'stage/runs/t217_rnd_preflight/summary.json').read_text());assert rnd['state']=='RND_PREFLIGHT_PASS'
    old=json.loads((ROOT/'stage/experiments/t216_left_specialist/protocol.json').read_text())
    plan={k:old[k] for k in ('checkpoint','retention_targets','primary_cases','task_snapshots','actor_lr','value_lr','clip_ratio','grad_clip','actor_attempts_per_round','minimum_objective_improvement','step_factors','max_behavior_mean_KL','batch_mean_tolerance','batch_logp_tolerance')}
    plan.update(experiment='T219',seed=2026103218,arms=['control','rnd'],max_rounds=4,per_arm_per_round=4,workers=4,
        wall_seconds=900,final_reserve_seconds=180,max_commands=120,max_physics=2200000,
        final_bytes=6*1024**2,peak_bytes=6*1024**2,export_reserve_bytes=5*1024**2,
        retention_weight=0.,max_parent_anchor_KL=1e9,max_balanced_anchor_MSE=1e9,
        rnd_checkpoint=rnd['checkpoint']['path'],rnd_scale=rnd['seen_error']['q95'],rnd_episode_cap=.05,
        rnd_predictor_updates_per_round=50,rnd_lr=.001,value_epochs=3,minibatch=256)
    plan['method_contract']=('Pairedcontrol/RNDsingleLEFTspecialist32freshtrainingepisodesmax, sameT194/full574210MLP/freshindependentV/nativeGaussian/seed/terminaldistance.3/MCbaseline/fullbatchguardedPPO. '
       'Botharms oldretentiondisabled, old7measurednotpromoted. ONLYrewarddifference RNDcap.05vs0;RNDmodelsbothwarm-startT217,fixednormalization/target/scale, '
       'eachbatchrewardcomputedbefore50predictorAdamupdates;collisionmasked,physicalcount/23334scaled. SameinitialV/RND/allactionsseeds;initialpairedobservationsmustmatch. '
       'NoCNN/GRU/maps/poseinRND/planner/waypoints/runtimeTeacher/Phase research. No novelty=GOALclaim ororiginalsceneedit. '
       'Everytrainingrolloutimmediatelyreplayedwithactualactions/exactsnapshot/behaviorpolicyINRAM, all10arraycanonicalhashes/finalintegrationbitwiseequalbeforelearning. '
       'Replaynottraininginteraction. Newlycapturedfilesretainactions,means,logp,counts,poses,flags,rewards;observationscaptured/hashcheckedinRAMandexactlyreconstructed, '
       'notwrittenthenremoved. Noexistingevidencedeleted. Per-roundpolicyhashes+source+initialstate+verifiedreplaydata+exactoptimizerrecipe reconstructallbehaviorpolicies. '
       'EachActor/V/RNDupdate independentlyrecomputedonshadowfromsameinitialstate/inputs/seeds;canonicalparameterANDoptimizerstatehashesequalaftereachbatch. '
       'Shadowupdatesseparatelycounted,neverusedtocontrolsimulator. No restoreacrosslibraryversionguarantee. '
       'KeepONLYfinalcompressedActorandRNDperarm,readbackhashverified,workerfinalold8 usesdeserializedweights. IntermediateweightsRAMonly,reconstructioncontractmandatory. '
       '900s,last180sevalreserve,6MiBown/global1GiB/free40GiB, no duplicate/resume/extension/nexttrialautolaunch,T159/T194unchanged.')
    plan['accounting_contract']=('Parentold8=8eval;4rounds*2arms*4=32freshtrainingrollouts+32immediatereplays;finalsavedreloadold8*2=16eval =>88commandsmax,hard120/2.2Mphysics. '
       'No per-roundold7gates;centerLEFTfinalonly, sampledGOALstopsafterpairedround thenfinaltests. No automatic54grid/promotion/longdistance. '
       'Atmost48Actorproposalsliveand48shadow;ValueandRNDupdatecountsexplicit. Diskguardreserves5MiBfinalmodels beforenewbatches. '
       'Ifreserve/wall/countfails stop,keepcompactfailures;finalincompletemissingnot0/8. Do not acceptnavigationconclusionsif replayortrainingreconstructionfails.')
    plan['references']=['https://arxiv.org/abs/1810.12894','https://github.com/openai/random-network-distillation/blob/master/policies/cnn_gru_policy_dynamics.py','https://arxiv.org/html/2404.12594']
    plan['runtime_correction']='T218 failed before physics/updates: missing ephemeral g5b dependency, independently reproduced. Verified original archive module now repository-local, no historical source changes. Four spawned native-loader handshakes mandatory before navigation. Same seed and scientific protocol; T218 remains ended, not resumed.'
    plan['references']+=['https://proceedings.mlr.press/v235/yang24w.html','https://docs.python.org/3/library/multiprocessing.html']
    files=list(HERE.glob('*.py'))+[Path(plan[k]) for k in ('checkpoint','retention_targets','rnd_checkpoint')]+[Path(x) for x in plan['task_snapshots']]+[
        ROOT/'stage/experiments/t216_left_specialist/support.py',ROOT/'stage/experiments/t216_left_specialist/learner.py',
        ROOT/'stage/experiments/t217_rnd_preflight/model.py',ROOT/'stage/experiments/native_update_guard.py',ROOT/'stage/experiments/t203_native_exploration/runtime.py',ROOT/'stage/experiments/t200_residual_placement/common.py']
    files += [HERE/'native_compat/g5b.py']
    plan['input_hashes']={str(p):sha(p) for p in files}
    used=sum(p.stat().st_size for p in (ROOT/'stage/runs').rglob('*') if p.is_file())
    assert used+plan['peak_bytes']<1024**3 and shutil.disk_usage(ROOT).free>40*1024**3
    OUT.mkdir()
    for p in (HERE/'protocol.json',OUT/'protocol.json'):p.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({'state':'REGISTERED','protocol_sha256':sha(HERE/'protocol.json'),'runs_MiB':used/1024**2}))
if __name__=='__main__':main()
