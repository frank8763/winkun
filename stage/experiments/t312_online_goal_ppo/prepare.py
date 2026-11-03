from ppo_learning import *
import ast

def main():
    assert not OUT.exists();p=read(ROOT/'stage/runs/t287_time_aligned_credit/protocol.json');base=read(ROOT/'stage/runs/t309_sequence_goal_value/protocol.json')
    protected=read(ROOT/'stage/runs/t310_initial_history_recovery/protocol.json')['input_hashes'];inputs=dict(protected)
    assert read(DOC/'T311_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T311.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    rows=read(ROOT/'stage/runs/t309_sequence_goal_value/cases.json');tasks=[];seeds={}
    for goal in [g for g in base['task_goals'] if g['split']=='train']:
        rr=[r for r in rows if r['original_job']['source_reset_id']==goal['id'] and r['arm']=='center'];rr.sort(key=lambda r:r['seed']);assert len(rr)==4
        j=copy.deepcopy(rr[0]['original_job']);j.pop('deadline_epoch',None);j.pop('expected_initial120_sha256',None);j.update(source_reset_id=goal['id'],goal_family=goal['id']);tasks.append(j)
        seeds[goal['id']]=[r['seed'] for r in rr[:2]]
        for r in rr[:2]:inputs[r['path']]=r['sha256']
    geometry=[dict(j,id='geometry_goal_'+str(k),source_reset_id=j['source_reset_id']) for k,j in enumerate(tasks)]
    geometry += [dict(j,id='geometry_'+j['id'],source_reset_id=j['id']) for j in p['development_jobs']+p['heldout_jobs']]
    assert len(tasks)==8 and len(geometry)==78 and len(p['old_development'])==211 and len(p['old_regression64'])==64 and len(p['acceptance_guard_jobs'])==20
    for f in [*HERE.glob('*.py'),DOC/'T312_PLAN.md',DOC/'T311_AUDIT.json',DOC/'T311_ENCODING_LEDGER.json',receipt,
        ROOT/'stage/experiments/goal_conditioned_ppo/advantage.py',ROOT/'stage/experiments/t309_sequence_goal_value/sequence_value.py',ROOT/'stage/experiments/t308_iid_native_goal_value/t308_core.py',ROOT/'stage/experiments/t280_local_cost/t280_worker.py',ROOT/'stage/experiments/t278_fixed_goal_starts/t278_guard.py']:
        inputs[str(f)]=C.sha(f)
    p={k:p[k] for k in ['checkpoint','checkpoint_sha256','initial_base','retention_states','retention_states_sha256','base_lr','memory_lr','retention_weight','max_conditional_k3','max_abs_log_ratio','old_development','old_regression64','old_centers','acceptance_guard_jobs','development_jobs','heldout_jobs','centers']}
    p.update(experiment='T312',policy_protocol=base['policy_protocol'],policy_hash=base['policy_hash'],input_hashes=inputs,
        arms=['mc','gae'],seed=2026110317,rounds=2,actor_steps=8,value_steps=500,value_lr=.001,**{'lambda':.95},
        training_tasks=tasks,goal_families=[j['source_reset_id'] for j in tasks],preflight_reference_seeds=seeds,geometry_jobs=geometry,
        max_commands=1172,max_physics=32816000,wall_seconds=7200,final_reserve_seconds=1800,audit_seconds=3600,
        final_bytes=43*1024**2,payload_bytes=int(37*1024**2),metadata_bytes=5*1024**2,
        training_episode_bytes=232*1024,old_compact_bytes=6*1024,new_compact_bytes=7*1024,actor_model_bytes=int(7.5*1024**2),
        global_bytes=1024**3,min_free_bytes=40*1024**3,
        contract='New bounded matched real-multigoal onpolicy neural stateV/MC versus SMDPGAE(.95) PPO pilot, not continuation ofT287/T291/T309 or novelmethodclaim. SafeT255 freshActorAdam both; original120hiddenresetActor/native2ms100hold/radius/reward unchanged. rho0 iid boundedGaussian sigma(.035,.03) matchesT309, no oldARconditional-logp reuse; this shared exploration/goaldata change means pilot is not a pure comparison to historicalAR MC. 8fixed actualgoals fromT309train L/R .75/1.5/3.5/6 with originalbase6 physicalinitials, goalalignedinitialhistory inbotharms andtruefactualhistorythereafter; finaloriginalold275/six54 useoriginalnativehistory. Goalconstant duringepisode, no runtimewaypoint/map/planner/teacher/router/Phase.2rounds x2arms x8goals x2draw=64live trajectories+64exactnative replays;first16twoarms exactsamepolicy/innovation trajectories physicallychargedtwice not32independentsamples. Afterpoliciesdiverge eacharm collectsitsowntrueonpolicy data. Eachbatch2fold wholedrawsplit:8completeepisodestrain8test, same120-64-64-1V withtrainfoldnormalizer,500Adam/fold,2fold; no queryQ or actionbackpropthroughV. EachVlive+shadow exact modelAdam/predictions; max4000liveV+4000shadowV. Heldout episode/goal-equalMSE<=.90zero and eachgoal<=1.10zero+1e-12, full48scoregradients/batch (16xzeroMC,MCminusV,GAE) allfinite/densityexact, pergoal2draw samplegradientvariance forMCV andGAE<=2zero+1e-12. These finitepilot screens notconfidencebound/safety/correctvalueproof. GAE_MCmean cosine reported notunbiasedtargetclaim,lambda<1biasacknowledged. BotharmsmustpassbatchscreensbeforeeitherActorupdates, failedVnotconnected,no model/lambda/seedgrid. SamefrozenzeroMC_RMSscale>=.05,no advantage centering; fresh trueiid marginalD.terms canonicalbatch1 density, PPOclip.2, same8fixedminibatch16rowsx16episodes=256 perround, ActorLRs/soft100 kept. NO PCGrad in eitherarm (shared implementation change vs historical, not GAEsinglefactorclaim); perstepfullbatch k3<=.02/maxlogratio1 viaexistingregisteredvalues, no hardactionMSE/max constraint. Eachblock8live+8shadow calls exact SHA; ifanynumericfail stopblock/rollbackActorAdam (no stepfactor grid), otherwise realold20allGOALaccept elsewholeblockrollback. Nativecacheonlysamepolicy hash, noindependentrepeatclaim. Max32liveActor+32shadow; rejected/intermediate modelstates retainedbyfulltrainingdata/exact deterministiceventrecipe+SHA andindependentaudit, not binarysnapshotperstep or deletion of existingmodels; final2ActorAdam+last2VAdam eachsaved/readback. Everyroundold8+six2centers reported distinct; finalbothold275+six54 always measured iftimebudgethealthy, allsafe then16fresh then78independentnative reload fullarray/integrationexact. No7/8training inpilot evenifpasses untilaudit/backup/newprotocol. Fulltraining16arrays120/history/innovations losslesschain<=232KiB,64full14.5MiB+4partials1MiB+2finalmodels15MiB+4Vevidence.5MiB+compact<=6MiB=37MiBpayload,metadata5MiB+1MiBbuffer=43MiB. 78zerophysicsgeometry+128collect/replay+80guard+40centers+658final+32fresh+156reload=1172commands;allmaxcap28k=32.816Mphysical;7200walllast1800final/audit3600zero-native max8064disposablehistoricalAdam. Preflight6syntheticdisposableAdam0native, NOT learningbenefit. Cleancommit/sourceinputSHA/fullrootremote/preflightrequired beforeunique launch; exclusivewriter nooutside runswrites/sourcesync whileactive. Endallparentworkersinhibitortimeoutabsent,independentarrays/history/density/source/model/Adam/optimizerrecipe/countaudit/fullrawRAMremote everymemberSHA then sourcesync. Preserveallfailures/currentmodels/videos;global1GiB/free40GiB, no expansion/restart/resume or automaticseedroundextension.')
    # Registered bound is conservative even when only one arm reaches the final gate.
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes']
    assert S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes'],('Global reserve',S.used(ROOT/'stage/runs'))
    for f,h in inputs.items():assert C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir();(OUT/'learning').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:C.save(f,p)
    print('T312 registered matched online MC/GAE pilot;0native/optimizer')
if __name__=='__main__':main()
