from t313_core import *
import ast

def main():
    assert not OUT.exists();old=ROOT/'stage/runs/t312_remaining_recovery';p=read(old/'protocol.json')
    assert read(old/'summary.json')['state']=='COMPLETE' and read(DOC/'T312_REMAINING_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T312R.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    inputs=dict(p['input_hashes'])
    # Old navigation clocks remain immutable in the referenced source protocol;
    # this separate zero-native diagnostic has no navigation learning window.
    for name in ['absolute_deadline_epoch','training_deadline_epoch','original_main_epoch','recovery_contract']:
        p.pop(name,None)
    for f in [*HERE.glob('*.py'),old/'protocol.json',old/'summary.json',old/'round_updates.json',DOC/'T312_REMAINING_AUDIT.json',DOC/'T313_PLAN.md',receipt]:inputs[str(f)]=K.C.sha(f)
    p.update(experiment='T313_OFFLINE_EMPIRICAL_FISHER_FEASIBILITY',input_hashes=inputs,fisher_rows_per_episode=8,fisher_damping=.1,cg_steps=10,cg_relative_squared_tolerance=1e-12,
        backtrack_steps=8,backtrack_decay=.5,objective_min_improvement=1e-10,goal_surrogate_tolerance=1e-10,wall_seconds=1800,audit_seconds=1800,
        final_bytes=14*1024**2,payload_bytes=12*1024**2,metadata_bytes=1024**2,model_bytes=3*1024**2,max_commands=0,max_physics=0,
        contract='New independent zero-native offline direction diagnostic, not continuation or budget extension of endedT312/R. First16 verified matching onpolicy multigoal trajectories/MCV/GAE/oldretention reused, no Vrefit/no newdata. Two original firststepAdam reconstructions discarded historical calls, not newnativeRL; exact originalcandidateActorAdam SHA mandatory. Capture clippeduphillgradient, samefixed256rows andseed2026110317. Empirical scoreouterproductF from first8 of same16draws each16episode,128with duplicates honestlycounted; fixeddamping.1/CG10/relative residualsquared1e-12, no analyticKL or TRPOguarantee. Both actualAdamdelta andFisherdirection normalized to SAME .02dampedempiricalquadratic then same8 bounded.5backtracking; raworiginalAdam result separatelyreported. Full16trajectory equalweight clippedsurrogate/physicaldiscountadv/commonoldMC_RMS/soft100 alloldanchors, originalk3<=.02/maxabslogratio<=1, fullobjectiveimprovement>1e-10 AND each6mgoal surrogatechange>=-1e-10. Twoadvantageestimators x2directions xatmost8=32offlineparameterproposals, neverdeployed; not0Actorparametermutations. Fourlastcandidatepolicies<=3MiBeach saved acceptedflag includingfailure, intermediateallhash/exactrecipe kept; no optimizer state substituted into safemodel. No seed/lambda/damping/CG/noise/threshold grids. No successrate or old275/six54/fresh16/reload78 inthis0native task. Feasibilitybotharms isonlyengineeringpilot; matchingAdamfeasible preventsclaimFisherbenefitjustfrombacktracking, no automatednewnative budget evenifpass. Independentall2historicalAdam/scorematrix/CG/trials/models exactreconstruction<=1800s thenRAMfullremoteeverymemberSHA/source. Main1800s/own14MiB/global1GiB/free40GiB, originalmodels/videos/failures remain, no Phase/map/planner/waypoint/runtimecriticteacherrouter; single120hiddenreset deploymentunchanged. Preflight0native0Adam0parameterproposal, syntheticSPDsolve+scalarFD+realrowdensity+parameterpackidentity. Cleancommitted exactremoterootbeforeunique main; allprocessesabsent beforeaudit/archive; no outsiderunswrites/source sync active.')
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    for f,h in inputs.items():assert K.C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:K.C.save(f,p)
    print('T313 bounded offline diagnostic registered,0native0Adam0parameterproposal')
if __name__=='__main__':main()
