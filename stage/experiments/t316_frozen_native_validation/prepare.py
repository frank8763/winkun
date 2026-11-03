from t316_core import *
import ast
def main():
    assert not OUT.exists();p=read(ROOT/'stage/runs/t315_full_trajectory_gradient/protocol.json');s=read(ROOT/'stage/runs/t315_full_trajectory_gradient/summary.json');m=read(ROOT/'stage/runs/t315_full_trajectory_gradient/metrics.json')
    assert s['state']=='COMPLETE' and s['empirical_fisher_viable_both_arms'] and s['Adam_viable_both_arms'] and read(DOC/'T315_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T315.json';assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    assert len(m['results'])==4 and all(x['accepted'] for x in m['results'])
    candidates=s['models'];inputs=dict(p['input_hashes'])
    for f in [*HERE.glob('*.py'),DOC/'T316_PLAN.md',DOC/'T315_AUDIT.json',receipt,ROOT/'stage/runs/t315_full_trajectory_gradient/summary.json',ROOT/'stage/runs/t315_full_trajectory_gradient/metrics.json',ROOT/'stage/experiments/t312_online_goal_ppo/ppo_worker.py',ROOT/'stage/experiments/t280_local_cost/t280_worker.py']:
        inputs[str(f)]=K.C.sha(f)
    for r in candidates.values():inputs[r['path']]=r['sha256']
    caps=[j['case']['step_cap'] for j in p['acceptance_guard_jobs']+p['development_jobs']]
    p.update(experiment='T316_FROZEN4_NATIVE_VALIDATION',input_hashes=inputs,candidates=candidates,max_commands=370,max_physics=4*sum(caps),preflight_seconds=300,wall_seconds=2400,final_reserve_seconds=90,audit_seconds=300,
        final_bytes=5*1024**2,payload_bytes=2*1024**2,metadata_bytes=int(2.5*1024**2),old_compact_bytes=6144,new_compact_bytes=7168,case_row_bytes=4096,
        contract='User explicitly continueafterT315. Separate finite frozen4 native validation, not optimizercontinuation/budgetextension ofT315. All4 fixed models acceptedjointoffline proxies/audit/fullbackup. Sameoriginalold20 andoriginalsix54 deterministicnativehistory/initialphysics/radius/collisionpriority/caps/single120hiddenresetActor;4x(20+54)=296native commands,74factorygeometry0physicspreflightchargedtotal370; maxphysical4sumcaps=7709372. Everycandidateold20 then six54 REGARDLESSold20fail, researchdiagnostic notpromotion; allrawfailurestays. No optimization/gradient/Adam(VorActor)/newseed/noise/threshold/LR/rounds/grid or runtimeCriticteacherplannerwaypointsmaproute,Phasefullyisolated. Only current4inputpolicies,notcopied; allmodelsimmutable/safeT255/T299/HDvideosuntouched. Nativeworkercanonicalbatch1deterministic originalt312/t280generate,sourcefrozen,role nevertraining. Strictbudget5MiB own/global1GiB/free40GiB; compact80oldx6KiB+216sixx7KiB=2039808B<=2MiBpayload; metadata2.5MiB(296rows<=4KiB each+protocol<=.5MiB+4workerlog64KiB+geometry/preflight/statusfailure/summary<=.5MiB),buffer.5MiB. 4compactarraysstored percase plusfull12arrays andterminalintegrationSHA,initial120/scene/goal/remainingphysicalfailures; intermediate120notstored/notclaimedreconstructedin0nativeaudit. fullarraysonlyRAM; no newvideos/largecheckpoints. main2400s last90forcleanup/audit3000native,external earlier strict; noextensionsorrestarts. Preflight74geometryallnoncontact/fourfrozenfirstactions/integrationunchanged/actualworstcompact/zeroAdamgradient. InputSHA/cleancommit/exactremoterootbeforeunique main. Activeexclusivewriter noother runswrite/protocolchanges/sourcesync. Endparentworkersinhibitortimeoutabsent independentall296compact/counts/flags/source/model/inputSHA/initialscenehashaudit thenfullrawRAMremoteeverymemberSHA/source. Old20allsafe+six54allsafe onlyscreen,notold275/fresh16/reload78pass;thoseNOTexecutednot0%failure. Noautomatic7/8ornewtrainingevenifscreenpass; fullsamecandidateold275/six54/fresh16/78/auditbackupneededbeforepromotion.')
    assert p['max_physics']==7709372
    assert 80*p['old_compact_bytes']+216*p['new_compact_bytes']<=p['payload_bytes']
    assert 296*p['case_row_bytes']+512*1024+4*65536+512*1024<=p['metadata_bytes']
    assert p['payload_bytes']+p['metadata_bytes']<p['final_bytes'];used=K.S.used(ROOT/'stage/runs');assert used+p['final_bytes']<=p['global_bytes'],(used,p['final_bytes'])
    for f,h in inputs.items():assert K.C.sha(f)==h,f
    for f in HERE.glob('*.py'):ast.parse(f.read_text())
    OUT.mkdir();(OUT/'episodes').mkdir()
    for f in [HERE/'protocol.json',OUT/'protocol.json']:K.C.save(f,p)
    print('T316 frozen4native validation registered; worstreserve',used+p['final_bytes'],'0native0Adam')
if __name__=='__main__':main()
