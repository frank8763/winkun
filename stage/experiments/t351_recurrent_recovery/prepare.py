from t351_core import *


def main():
    assert not OUT.exists();assert read(DOC/'T350_AUDIT.json')['state']=='PASS'
    receipt=ROOT/'stage/docs/long_distance_20261101/release_receipt_T350.json'
    assert read(receipt)['state']=='FULL_REMOTE_READBACK_AND_EVERY_MEMBER_SHA_VERIFIED'
    p=copy.deepcopy(read(ROOT/'stage/experiments/t349_consistent_demo/protocol.json'))
    long=read(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json')
    for key in ['final_old_jobs','six_jobs','fresh_jobs','reload_jobs']:p[key]=long[key]
    p.update(experiment='T351_CURRENT_HISTORY_RECURRENT_RL_INDUCED_OLD_REFERENCE',
        absolute_stop_epoch=1793836800.,
        historical_cases=str(ROOT/'stage/runs/t349_consistent_demo/cases.jsonl'),
        preflight_sequence=str(ROOT/'stage/runs/t347_original_start_human/attempt_01.npz'),
        initial_model=str(ROOT/'stage/runs/t349_consistent_demo/memory_final.pt.gz'),arms=['memory'],
        sequence_sources=[],raw_sigma=[.20,.20],actor_lr=.0003,epochs=4,rounds=2,ppo_clip=.1,
        old_weight=1.,induced_weight=1.,support_weight=1.,seed=20261104351,
        wall_seconds=5400,final_reserve_seconds=1800,audit_seconds=1200,all_work_seconds=9000,
        final_bytes=int(16.5*1024**2),old_full_bytes=256*1024,full_bytes=512*1024,
        max_old_full=20,max_new_full=10,partial_bytes=512*1024,model_bytes=256*1024,
        compact_bytes=4096,max_compact_cases=700,metadata_bytes=2*1024**2,logs_bytes=128*1024,
        max_commands=1000,max_physics=40_000_000,per_batch_admission_seconds=60,
        preflight_discarded_Adam=4,
        contract='Evening bounded six-metre job,NOT overnight. T349memory+freshAdam; same6306learnedparameters/120/GRU16 actualcarry and T255 frozenbase. New fullcurrenthistory clippedGaussianPPO plus training-only T255 oldreference at real ownold20 visitedstates and two ownsuccessfulsixtrajectories; jointengineeringpilot not causal module ablation/novel algorithm. IID fixedrawsigma.2 both,actualfloat32edge tailmasses,unclippedlocation policymean; notoldAR/tanh or calibratedexecutedvariance. Two rounds: four exploratory originalstart trajectories from wide_dev2 then wide_dev4, each exactnative replay, max8live8shadowAdam. FullG0 peerLOO constant advantages, fixedscale1/400; PPO surrogate+old211/inducedold20/selfsuccess weights1 each, noCritic or fakehumanlogp. Reconstruct wholecurrentsequence from realepisodezero, causal BPTT nohidden replay. Initialold275 fullcount to identify preexisting failure, old20full saved forreference, initialwide5 expectedT3492/5 and original2, two successfulwide fullnative reconstruction+exactreplay historicalnotnewdiscovery. Accept each complete round only if old20 and wide5 prior successfulIDsets are supersets; rejection/interrupt entireActorAdamrollback. Researchretention notsafetycertificate; T255notreplaced. After2rejects stop, noseed/epoch/noisegrid. Finalold20/wide5/original2 and27diskmodelnativeexactreadbacks; allsafe first then completeold275/six54, allsafe then16unconsumedfresh+78independentnativeexactmodelreadback. Missingnull, no7/8/overnight unless allfullgates auditbackup actuallypass and independentnextprotocol. Samewheel5force120/wide1.17m80s/narrow.57m56s. All1000commands40Mphysical/5400s last1800final/audit1200/all9000/own16.5MiB/global1GiB/free40GiB; completeworstreserved/singlewriter, nooutsidewrites orsource sync duringnative. Driverunique finalizer afterallnativeexit, completeoriginalRAMbackup everymemberSHA/source tree. Allfailures preserved.')
    assert 20*p['old_full_bytes']+10*p['full_bytes']+2*p['partial_bytes']+2*p['model_bytes']+700*p['compact_bytes']+p['metadata_bytes']+p['logs_bytes']<=p['final_bytes']
    files=set(p['input_hashes'])|{str(f) for f in HERE.glob('*.py')}|{str(DOC/'T351_PLAN.md'),str(DOC/'T350_AUDIT.json'),str(receipt),
        p['initial_model'],p['historical_cases'],p['preflight_sequence'],str(ROOT/'stage/experiments/t349_consistent_demo/protocol.json'),str(ROOT/'stage/experiments/t340_demo_supported_recovery/protocol.json')}
    p['input_hashes']={f:K.C.sha(f) for f in sorted(files)}
    assert K.S.used(ROOT/'stage/runs')+p['final_bytes']<=p['global_bytes']
    OUT.mkdir();(OUT/'episodes').mkdir();save(HERE/'protocol.json',p);save(OUT/'protocol.json',p)
    print(json.dumps(dict(state='REGISTERED',reserved_global=K.S.used(ROOT/'stage/runs')-K.S.used(OUT)+p['final_bytes'],native=0,Adam=0)))


if __name__=='__main__':main()
